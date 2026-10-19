/* r3d_band.hip -- the four device-band ops the MiniCPM5-2B bf16 graph declares and no
 * other loadable plugin covers on RDNA3: quant_act_fp8, gated_quant_fp8, add, gather_rows.
 *
 * These are the ops the planner cannot push to a host kernel (weightless, device-band),
 * so until this file exists a bf16 trunk that quantises its MLP activations will not
 * start at all on this card. The arithmetic is libref's (ref_quant.cpp /
 * ref_elementwise.cpp, Apache-2.0, (c) radiance authors -- ported with attribution):
 *
 *   * the 8-bit group: amax over the group (plain strict-max, NaN-skipping like fabs(x) >
 *     amax), scale = amax * (1/448) -- a DEQUANT multiplier, 1.0 for an all-zero group --
 *     codes = clamp(v * (1/scale), +-448) encoded by rad_f32_to_fp8e4m3, the same ABI
 *     inline libref's st_dt calls;
 *   * gated: sigmoid in the double-rounded-exp form (ref_exp's glibc rule), the gated
 *     value narrowed to bf16 ONCE and re-widened for the quantiser, optional out_bf16
 *     fed the same narrowed value; the gate is recomputed per pass rather than staged --
 *     deterministic, so the two passes agree bit for bit and no row ever outgrows shared;
 *   * add: the double-rounded `div` form (y = rd(rd(a+b)/div)); b broadcasts along rows
 *     ONLY at exactly n elements;
 *   * gather_rows: negative index writes a zero row (the padding convention). One
 *     documented deviation: an out-of-range index >= src_rows zeroes the row instead of
 *     failing the launch -- the reference pre-validates on the host, which a device
 *     kernel cannot, and a zero row is what the padding convention already means.
 *
 * fp8 is storage here, not arithmetic: nothing in this file needs E4M3 hardware, only
 * the encoder, so RDNA3 serves the quantising trunk exactly as the oracle defines it.
 * (The GEMMs consuming these codes run on the host reference today; a device fp8 GEMM
 * with software decode is the follow-up that restores MLP speed.)
 */
#include <hip/hip_runtime.h>
#include <cstring>
#include <cmath>

#include "rad_abi.h"
#include "rad_plugin.h"

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kWarps   = kThreads / 32;

__device__ __forceinline__ float r3d_exp_f(float x) { return (float)exp((double)x); }
__device__ __forceinline__ float r3d_sigmoid(float x) { return 1.0f / (1.0f + r3d_exp_f(-x)); }

/* Device mirrors of rad_plugin.h's host-inline conversions, bit for bit (same pattern as
 * r3d_rmsnorm.hip's): the bf16 pair first, then the e4m3 encoder verbatim in spirit --
 * RTNE, subnormal shift, saturate at 448 -- since the fp8 codes this file writes are read
 * by the reference GEMMs and must be the codes the oracle would have written. */
__device__ __forceinline__ float r3d_bf16_to_f32(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}
__device__ __forceinline__ uint16_t r3d_f32_to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ uint8_t r3d_f32_to_e4m3(float f) {
    const uint32_t u = __float_as_uint(f);
    const uint8_t  s = (uint8_t)((u >> 24) & 0x80u);
    const uint32_t a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return (uint8_t)(s | 0x7fu);     /* nan */
    if (a >= 0x43e00000u) return (uint8_t)(s | 0x7eu);    /* >= 448: clamp to the largest */
    const int e = (int)(a >> 23) - 127;
    uint32_t m = a & 0x7fffffu;
    if (e < -10) return s;                                /* below half the smallest subnormal */
    if (e < -6) {                                         /* subnormal fp8 */
        const int sh = -6 - e + 20;
        m |= 0x800000u;
        const uint32_t lo = m & ((1u << sh) - 1u);
        uint32_t q = m >> sh;
        const uint32_t half = 1u << (sh - 1);
        if (lo > half || (lo == half && (q & 1u))) ++q;
        return (uint8_t)(s | q);
    }
    uint32_t q = m >> 20;
    const uint32_t lo = m & 0xfffffu;
    int ee;
    if (lo > 0x80000u || (lo == 0x80000u && (q & 1u))) ++q;
    ee = e + 7;
    if (q == 0x8u) { q = 0; ++ee; }
    if (ee > 15 || (ee == 15 && q >= 7u)) return (uint8_t)(s | 0x7eu);
    return (uint8_t)(s | ((uint32_t)ee << 3) | q);
}

/* ldt/stt at the dtypes the engine bands carry: f32, bf16. Anything else is refused at
 * the entry, so these helpers need no other cases. */
__device__ __forceinline__ float ldt2(const uint8_t* p, uint32_t dt, int64_t i) {
    return dt == RAD_F32 ? ((const float*)p)[i]
                         : r3d_bf16_to_f32(((const uint16_t*)p)[i]);
}
__device__ __forceinline__ void stt2(uint8_t* p, uint32_t dt, int64_t i, float v) {
    if (dt == RAD_F32) ((float*)p)[i] = v;
    else ((uint16_t*)p)[i] = r3d_f32_to_bf16(v);
}

/* Block reduce: maximum of per-thread partials. fmaxf skips NaN exactly like the ref's
 * strict `>` scan (NaN never wins; the all-NaN case keeps the initial 0, as ref's does). */
__device__ __forceinline__ float block_max(float v, float* s_part) {
    for (uint32_t off = 16; off; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xFFFFFFFFull, v, off));
    const uint32_t lane = threadIdx.x & 31u, wid = threadIdx.x >> 5;
    if (lane == 0) s_part[wid] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float r = s_part[0];
        for (uint32_t w = 1; w < kWarps; ++w) r = fmaxf(r, s_part[w]);
        s_part[0] = r;
    }
    __syncthreads();
    return s_part[0];
}

/* ------------------------------------------------------------------ quant_act_fp8
 * One block per (row, group): plain row-major codes, group scale = amax/448. */
__global__ __launch_bounds__(kThreads) void quant_act_fp8_kernel(
    const uint8_t* __restrict__ x, uint32_t dt, int64_t x_pitch,
    uint8_t* __restrict__ q, int64_t q_pitch,
    float* __restrict__ scale, int64_t s_pitch,
    int64_t n, int64_t g, int64_t ngrp) {
    __shared__ float s_part[kWarps]; __shared__ float s_sc;
    const int64_t row  = blockIdx.x / ngrp;
    const int64_t gi   = blockIdx.x % ngrp;
    const int64_t k0   = gi * g, k1 = (k0 + g < n) ? k0 + g : n;
    const uint8_t* __restrict__ xr = x + row * x_pitch;
    uint8_t* __restrict__ qr = q + row * q_pitch;

    float amax = 0.0f;
    for (int64_t i = (int64_t)threadIdx.x; i < k1 - k0; i += (int64_t)kThreads) {
        const float m = fabsf(ldt2(xr, dt, k0 + i));
        if (m > amax) amax = m;   /* strict > matches ref; fabsf(NaN) never wins */
    }
    amax = block_max(amax, s_part);
    if (threadIdx.x == 0) {
        const float sc = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
        s_sc = sc;
        scale[row * s_pitch + gi] = sc;
    }
    __syncthreads();
    const float inv = 1.0f / s_sc;
    for (int64_t i = (int64_t)threadIdx.x; i < k1 - k0; i += (int64_t)kThreads) {
        float c = ldt2(xr, dt, k0 + i) * inv;
        if (c >  448.0f) c =  448.0f;
        if (c < -448.0f) c = -448.0f;
        qr[k0 + i] = r3d_f32_to_e4m3(c);
    }
}

/* ------------------------------------------------------------------ gated_quant_fp8
 * One block per row; per group: recompute the gated bf16 value for the amax pass, then
 * again for the encode pass (deterministic -- identical bits, so no row-sized staging). */
__device__ __forceinline__ float gated_h(const uint8_t* __restrict__ gub, uint32_t dt,
                                         int64_t gs, int64_t n, int64_t i,
                                         int silu) {
    const float gt = ldt2(gub, dt, i * gs);
    const float up = ldt2(gub, dt, (n + i) * gs);
    const float sg = r3d_sigmoid(gt);
    return r3d_bf16_to_f32(r3d_f32_to_bf16((silu ? gt * sg : sg) * up));
}

__global__ __launch_bounds__(kThreads) void gated_quant_fp8_kernel(
    const uint8_t* __restrict__ gu, uint32_t dt, int64_t gu_pitch, int64_t gs,
    uint8_t* __restrict__ q, int64_t q_pitch,
    float* __restrict__ scale, int64_t s_pitch,
    uint8_t* __restrict__ ob, uint32_t ob_dt, int64_t ob_pitch,
    int64_t n, int64_t g, int64_t ngrp, int silu) {
    __shared__ float s_part[kWarps]; __shared__ float s_sc;
    const int64_t row = blockIdx.x;
    const uint8_t* __restrict__ gub = gu + row * gu_pitch;
    uint8_t* __restrict__ qr = q + row * q_pitch;
    uint8_t* __restrict__ yr = ob ? ob + row * ob_pitch : nullptr;

    for (int64_t gi = 0; gi < ngrp; ++gi) {
        const int64_t k0 = gi * g, k1 = (k0 + g < n) ? k0 + g : n;
        float amax = 0.0f;
        for (int64_t i = (int64_t)threadIdx.x; i < k1 - k0; i += (int64_t)kThreads) {
            const float m = fabsf(gated_h(gub, dt, gs, n, k0 + i, silu));
            if (m > amax) amax = m;
        }
        amax = block_max(amax, s_part);
        if (threadIdx.x == 0) {
            const float sc = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
            s_sc = sc;
            scale[row * s_pitch + gi] = sc;
        }
        __syncthreads();
        const float inv = 1.0f / s_sc;
        for (int64_t i = (int64_t)threadIdx.x; i < k1 - k0; i += (int64_t)kThreads) {
            const float h = gated_h(gub, dt, gs, n, k0 + i, silu);
            if (yr) stt2(yr, ob_dt, k0 + i, h);
            float c = h * inv;
            if (c >  448.0f) c =  448.0f;
            if (c < -448.0f) c = -448.0f;
            qr[k0 + i] = r3d_f32_to_e4m3(c);
        }
        __syncthreads();   /* s_sc reused next group */
    }
}

/* ------------------------------------------------------------------ add */
__global__ __launch_bounds__(kThreads) void add_kernel(
    const uint8_t* __restrict__ A, uint32_t dtA, int64_t a_pitch, int64_t as,
    const uint8_t* __restrict__ B, uint32_t dtB, int64_t b_pitch, int64_t bs,
    uint8_t* __restrict__ Y, uint32_t dtY, int64_t y_pitch, int64_t ys,
    int64_t n, float div, int use_div) {
    const int64_t r = blockIdx.x;
    const uint8_t* __restrict__ ab = A + r * a_pitch;
    const uint8_t* __restrict__ bb = B + r * b_pitch;
    uint8_t* __restrict__ yb = Y + r * y_pitch;
    for (int64_t i = (int64_t)threadIdx.x; i < n; i += (int64_t)kThreads) {
        const float u = ldt2(ab, dtA, i * as);
        const float v = ldt2(bb, dtB, i * bs);
        float s = u + v;
        if (use_div) {
            if (dtY == RAD_F32) s = s / div;
            else s = r3d_bf16_to_f32(r3d_f32_to_bf16(s)) / div;
        }
        stt2(yb, dtY, i * ys, s);
    }
}

/* ------------------------------------------------------------------ gather_rows */
__global__ __launch_bounds__(kThreads) void gather_rows_kernel(
    const uint8_t* __restrict__ x, uint32_t dtX, int64_t x_pitch, int64_t xs,
    const int32_t* __restrict__ idx,
    uint8_t* __restrict__ y, uint32_t dtY, int64_t y_pitch, int64_t ys,
    int64_t n, int64_t src_rows) {
    const int64_t m = blockIdx.x;
    const int64_t r = idx[m];
    uint8_t* __restrict__ yb = y + m * y_pitch;
    if (r < 0 || r >= src_rows) {   /* padding convention; OOR documented at the entry */
        for (int64_t i = (int64_t)threadIdx.x; i < n; i += (int64_t)kThreads)
            stt2(yb, dtY, i * ys, 0.0f);
        return;
    }
    const uint8_t* __restrict__ xb = x + r * x_pitch;
    for (int64_t i = (int64_t)threadIdx.x; i < n; i += (int64_t)kThreads)
        stt2(yb, dtY, i * ys, ldt2(xb, dtX, i * xs));
}

/* ------------------------------------------------------------------ entry helpers */
static inline bool dt_act_ok(uint32_t dt) { return dt == RAD_F32 || dt == RAD_BF16; }
/* Row-major [rows, cols] reader, rank-2 with a unit last stride, OR rank-1 single row
 * (pitch 0 -- the reference's rowoff walks a rank-1 tensor to offset 0 for every row, so
 * a broadcast `b` arrives as a bare [n] vector; requiring rank 2 refused those cases). */
static inline int row2_pitch(const RadTensor* t, int64_t cols, int64_t* pitch) {
    if (!t || !t->data) return -1;
    if (t->rank == 1) {
        if (rad_tensor_numel(t) != cols || t->stride[0] != 1) return -2;
        *pitch = 0;
        return 0;
    }
    if (t->rank != 2 || t->shape[1] != cols || t->stride[1] != 1) return -3;
    *pitch = t->stride[0];
    return 0;
}
static inline int64_t param_i(const RadArgs* a, const char* k, int64_t fb) {
    long long v;
    return rad_args_geti(a, k, &v) ? (int64_t)v : fb;
}

} /* anonymous namespace */

/* ------------------------------------------------------------------ entries */
extern "C" int r3d_rad_quant_act_fp8(const RadArgs* a, RadStream s) {
    const RadTensor* x = rad_arg_in(a, 0);
    const RadTensor* q = rad_arg_in(a, 1);
    const RadTensor* sc = rad_arg_in(a, 2);
    if (!x || !q || !sc) return RAD_E_INVAL;
    if (!dt_act_ok(x->dtype)) return RAD_E_DTYPE;
    if (q->dtype != RAD_F8E4M3 || sc->dtype != RAD_F32) return RAD_E_DTYPE;
    const int64_t n = param_i(a, "n", x->rank >= 2 ? x->shape[1] : 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rad_tensor_numel(x) / n;
    if (rows <= 0 || rows > 0xffff) return RAD_E_SHAPE;
    int64_t g = param_i(a, "group", n);
    if (g <= 0 || g > n) g = n;
    const int64_t ngrp = (n + g - 1) / g;
    if (rad_tensor_numel(q) < rows * n || rad_tensor_numel(sc) < rows * ngrp) return RAD_E_SHAPE;
    int64_t xp, qp, sp;
    if (row2_pitch(x, n, &xp)) return RAD_E_STRIDE;
    if (row2_pitch(q, n, &qp)) return RAD_E_STRIDE;
    if (row2_pitch(sc, ngrp, &sp)) return RAD_E_STRIDE;
    quant_act_fp8_kernel<<<dim3((uint32_t)(rows * ngrp)), dim3(kThreads), 0,
                           (hipStream_t)(uintptr_t)s>>>(
        (const uint8_t*)x->data, x->dtype, rad_dtype_bytes(x->dtype, xp), (uint8_t*)q->data, qp,
        (float*)sc->data, sp, n, g, ngrp);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r3d_rad_gated_quant_fp8(const RadArgs* a, RadStream s) {
    const RadTensor* gu = rad_arg_in(a, 0);
    const RadTensor* q  = rad_arg_in(a, 1);
    const RadTensor* sc = rad_arg_in(a, 2);
    const RadTensor* ob = (a->n_t > 3 && a->t[3].data) ? &a->t[3] : nullptr;
    if (!gu || !q || !sc) return RAD_E_INVAL;
    if (!dt_act_ok(gu->dtype)) return RAD_E_DTYPE;
    if (q->dtype != RAD_F8E4M3 || sc->dtype != RAD_F32) return RAD_E_DTYPE;
    if (ob && !dt_act_ok(ob->dtype)) return RAD_E_DTYPE;
    const int64_t n = param_i(a, "n", gu->rank >= 2 ? gu->shape[1] / 2 : 0);
    if (n <= 0) return RAD_E_SHAPE;
    const char* act = rad_args_gets(a, "act");
    const int silu = !act || strcmp(act, "sigmoid") != 0;   /* absent or "silu" -> gate*up */
    int64_t g = param_i(a, "group", n);
    if (g <= 0 || g > n) g = n;
    const int64_t ngrp = (n + g - 1) / g;
    const int64_t rows = rad_tensor_numel(gu) / (2 * n);
    if (rows <= 0 || rows > 0xffff) return RAD_E_SHAPE;
    if (rad_tensor_numel(q) < rows * n || rad_tensor_numel(sc) < rows * ngrp) return RAD_E_SHAPE;
    if (ob && rad_tensor_numel(ob) < rows * n) return RAD_E_SHAPE;
    int64_t gp, qp, sp, op = 0;
    if (row2_pitch(gu, 2 * n, &gp)) return RAD_E_STRIDE;
    if (row2_pitch(q, n, &qp)) return RAD_E_STRIDE;
    if (row2_pitch(sc, ngrp, &sp)) return RAD_E_STRIDE;
    if (ob) { if (row2_pitch(ob, n, &op)) return RAD_E_STRIDE; }
    gated_quant_fp8_kernel<<<dim3((uint32_t)rows), dim3(kThreads), 0,
                             (hipStream_t)(uintptr_t)s>>>(
        (const uint8_t*)gu->data, gu->dtype, rad_dtype_bytes(gu->dtype, gp), 1,
        (uint8_t*)q->data, qp, (float*)sc->data, sp,
        ob ? (uint8_t*)ob->data : nullptr, ob ? ob->dtype : RAD_F32,
        ob ? rad_dtype_bytes(ob->dtype, op) : 0,
        n, g, ngrp, silu);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r3d_rad_add(const RadArgs* a, RadStream s) {
    const RadTensor* A = rad_arg_in(a, 0);
    const RadTensor* B = rad_arg_in(a, 1);
    const RadTensor* Y = rad_arg_in(a, 2);
    if (!A || !B || !Y) return RAD_E_INVAL;
    if (!dt_act_ok(A->dtype) || !dt_act_ok(B->dtype) || !dt_act_ok(Y->dtype)) return RAD_E_DTYPE;
    const int64_t n = param_i(a, "n", rad_tensor_numel(Y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rad_tensor_numel(Y) / n;
    if (rows <= 0 || rows > 0xffff) return RAD_E_SHAPE;
    if (rad_tensor_numel(A) < rows * n) return RAD_E_SHAPE;
    const bool bcast = rad_tensor_numel(B) == n && rows > 1;
    if (!bcast && rad_tensor_numel(B) < rows * n) return RAD_E_SHAPE;
    const double dv = rad_args_getf_or(a, "div", 1.0);   /* ref: reject 0 and non-finite */
    if (!(dv != 0.0) || !std::isfinite(dv)) return RAD_E_INVAL;
    const float div = (float)dv;
    const int use_div = div != 1.0f;
    int64_t ap = 0, bp = 0, yp = 0;
    if (row2_pitch(A, n, &ap)) return RAD_E_STRIDE;
    if (row2_pitch(Y, n, &yp)) return RAD_E_STRIDE;
    if (bcast) { if (row2_pitch(B, n, &bp)) return RAD_E_STRIDE; bp = 0; }
    else if (row2_pitch(B, n, &bp)) return RAD_E_STRIDE;
    add_kernel<<<dim3((uint32_t)rows), dim3(kThreads), 0, (hipStream_t)(uintptr_t)s>>>(
        (const uint8_t*)A->data, A->dtype, rad_dtype_bytes(A->dtype, ap), 1,
        (const uint8_t*)B->data, B->dtype, rad_dtype_bytes(B->dtype, bp), 1,
        (uint8_t*)Y->data, Y->dtype, rad_dtype_bytes(Y->dtype, yp), 1,
        n, div, use_div);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r3d_rad_gather_rows(const RadArgs* a, RadStream s) {
    const RadTensor* x = rad_arg_in(a, 0);
    const RadTensor* idx = rad_arg_in(a, 1);
    const RadTensor* y = rad_arg_in(a, 2);
    if (!x || !idx || !y) return RAD_E_INVAL;
    if (!dt_act_ok(x->dtype) || y->dtype != x->dtype) return RAD_E_DTYPE;
    if (idx->dtype != RAD_I32) return RAD_E_DTYPE;
    const int64_t n = param_i(a, "n", y->rank >= 2 ? y->shape[1] : 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rad_tensor_numel(y) / n;
    const int64_t src_rows = rad_tensor_numel(x) / n;
    if (rows <= 0 || rows > 0xffff || src_rows <= 0) return RAD_E_SHAPE;
    if (rad_tensor_numel(idx) < rows) return RAD_E_SHAPE;
    if (idx->rank != 1 || idx->stride[0] != 1) return RAD_E_SHAPE;
    int64_t xp, yp;
    if (row2_pitch(x, n, &xp)) return RAD_E_STRIDE;
    if (row2_pitch(y, n, &yp)) return RAD_E_STRIDE;
    gather_rows_kernel<<<dim3((uint32_t)rows), dim3(kThreads), 0, (hipStream_t)(uintptr_t)s>>>(
        (const uint8_t*)x->data, x->dtype, rad_dtype_bytes(x->dtype, xp), 1,
        (const int32_t*)idx->data,
        (uint8_t*)y->data, y->dtype, rad_dtype_bytes(y->dtype, yp), 1, n, src_rows);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
