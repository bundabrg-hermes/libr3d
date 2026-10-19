/* r3d_more.hip -- the three device bands the MiniCPM5-2B bf16 graph declares that no
 * ported row covers: silu_mul, quant_act_i8, row_topk. The engine names them at resolve
 * and they are weightless, so a bf16 trunk carrying them cannot START on RDNA3 until
 * these exist (the planner has no host fallback for a band without weights to place).
 *
 * The arithmetic is libref's (ref_elementwise.cpp / ref_quant.cpp / ref_moe.cpp,
 * Apache-2.0, (c) radiance authors -- ported with attribution):
 *
 *   * silu_mul: y[.., i] = silu(gate_up[.., i]) * gate_up[.., n + i], the gate the FIRST
 *     half. Unlike act_mul there is NO rounding of the activation before the product:
 *     silu, the product, and the narrowing are one f32 expression chain, and the sigmoid
 *     is the double-rounded-exp form (ref_exp's glibc rule) exactly as r3d_actmul.hip's.
 *   * quant_act_i8: one scale per row, max(amax, 1e-8)/127, codes x*(1/scale) rounded
 *     nearest-even (NaN -> 0) and stored at i8_step_pos(c) -- the WMMA A-fragment byte
 *     order, bits 2 and 3 of c swapped, the map its own inverse. The optional fourth
 *     operand is the `_asum` form: per-group code sums, group-major [n/g][Mpad] with
 *     Mpad = ceil(M/16)*16; rows M..Mpad-1 are not written.
 *   * row_topk: exact per-row top-R, descending, ties to the LOWER column -- one round
 *     of block argmax per selected element under the same (value desc, id asc) total
 *     order the sample chain uses, which is topk_next's window verbatim: a candidate
 *     must not beat the previous pick and must not be the previous pick or earlier.
 *     Unfilled slots report id -1 (no vocab_off shift on the -1, as ref) and -infinity.
 *
 * The kernels are correctness-first: row_topk re-reads the row R times (L2-resident at
 * the vocab widths the graph carries), and quant_act_i8 keeps the two-pass row. Speed
 * rows are a follow-up; the bands' first duty is making the graph start.
 */
#include <hip/hip_runtime.h>
#include <cstring>
#include <cmath>

#include "rad_abi.h"
#include "rad_plugin.h"

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kWarps   = kThreads / 32;

/* Device mirrors of the host-inline conversions, same pattern as r3d_band.hip. */
__device__ __forceinline__ float r3d_bf16_to_f32(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}
__device__ __forceinline__ uint16_t r3d_f32_to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float ldt2(const uint8_t* p, uint32_t dt, int64_t i) {
    return dt == RAD_F32 ? ((const float*)p)[i]
                         : r3d_bf16_to_f32(((const uint16_t*)p)[i]);
}
__device__ __forceinline__ float r3d_exp_f(float x) { return (float)exp((double)x); }
__device__ __forceinline__ float r3d_sigmoid(float x) { return 1.0f / (1.0f + r3d_exp_f(-x)); }

/* sat_round's i8 arm: NaN to zero, RTNE (device rintf is round-to-nearest-even), then
 * clamp to [-128, 127]. ref's st_dt(RAD_I8, ...) with the same three steps. */
__device__ __forceinline__ int8_t r3d_i8_sat(float v) {
    if (v != v) return 0;
    const float r = rintf(v);
    return (int8_t)(r < -128.0f ? -128.0f : (r > 127.0f ? 127.0f : r));
}

/* i8_step_pos(c): bits 2 and 3 of c swapped -- ref_common.h, its own inverse. */
__device__ __forceinline__ int64_t r3d_i8_step_pos(int64_t c) {
    return (c & ~(int64_t)12) | ((c & 4) << 1) | ((c & 8) >> 1);
}

/* Block reduce: maximum of per-thread partials (same as r3d_band.hip's). */
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

/* The one total order the engine's selections use: value descending, ties to the lower
 * column (r3d_sample.hip's rule, topk_next's rule, identical). */
__device__ __forceinline__ bool r3d_beats(float av, int64_t ai, float bv, int64_t bi) {
    return bi < 0 || av > bv || (av == bv && ai < bi);
}

/* ------------------------------------------------------------------ silu_mul
 * Flat over rows*n: one thread an output element. The gate is the first half of the
 * [M, 2n] row; the f32 product narrows to y's dtype once. */
__global__ __launch_bounds__(kThreads) void silu_mul_kernel(
    const uint8_t* __restrict__ gu, uint32_t dt, int64_t gu_pitch,
    uint8_t* __restrict__ y, int64_t y_pitch,
    int64_t n, int64_t total) {
    for (int64_t idx = (int64_t)blockIdx.x * kThreads + threadIdx.x; idx < total;
         idx += (int64_t)gridDim.x * kThreads) {
        const int64_t r = idx / n, i = idx - r * n;
        const uint8_t* __restrict__ gb = gu + r * gu_pitch;
        const float g = ldt2(gb, dt, i);
        const float up = ldt2(gb, dt, n + i);
        const float silu = g * r3d_sigmoid(g);           /* ref: x * 1/(1+exp(-x)) */
        if (dt == RAD_F32) ((float*)(y + r * y_pitch))[i] = silu * up;
        else ((uint16_t*)(y + r * y_pitch))[i] = r3d_f32_to_bf16(silu * up);
    }
}

/* ------------------------------------------------------------------ quant_act_i8
 * One block a row: amax pass, scale broadcast, code pass in the step order, and the
 * optional group-sum pass over the codes this block just wrote. */
__global__ __launch_bounds__(kThreads) void quant_act_i8_kernel(
    const uint8_t* __restrict__ x, uint32_t dt, int64_t x_pitch,
    int8_t* __restrict__ q, int64_t q_pitch,
    float* __restrict__ scale,
    float* __restrict__ asum, int64_t asum_pitch,
    int64_t n, int64_t g) {
    __shared__ float s_part[kWarps]; __shared__ float s_inv;
    const int64_t row = blockIdx.x;
    const uint8_t* __restrict__ xr = x + row * x_pitch;
    int8_t* __restrict__ qr = q + row * q_pitch;

    float amax = 0.0f;
    for (int64_t i = (int64_t)threadIdx.x; i < n; i += (int64_t)kThreads) {
        const float m = fabsf(ldt2(xr, dt, i));
        if (m > amax) amax = m;   /* strict > == ref's std::max: NaN never wins */
    }
    amax = block_max(amax, s_part);
    if (threadIdx.x == 0) {
        const float s0 = (amax < 1e-8f) ? 1e-8f : amax;   /* std::max(amax, 1e-8) */
        const float sc = s0 / 127.0f;
        scale[row] = sc;
        s_inv = 1.0f / sc;
    }
    __syncthreads();
    const float inv = s_inv;
    for (int64_t i = (int64_t)threadIdx.x; i < n; i += (int64_t)kThreads)
        qr[r3d_i8_step_pos(i)] = r3d_i8_sat(ldt2(xr, dt, i) * inv);

    if (!asum || g <= 0) return;
    __syncthreads();   /* the sums read codes every thread wrote */
    for (int64_t gi = (int64_t)threadIdx.x; gi < n / g; gi += (int64_t)kThreads) {
        int64_t sum = 0;
        for (int64_t i = gi * g; i < (gi + 1) * g; ++i) sum += qr[r3d_i8_step_pos(i)];
        asum[gi * asum_pitch + row] = (float)sum;
    }
}

/* ------------------------------------------------------------------ row_topk
 * One block a row, R rounds of block argmax with the after-the-previous-pick window.
 * The row is read through ldt2 (the constraint says bf16; f32 works too for cold
 * callers). The window and the comparison rule are topk_next's exactly, so the
 * selection -- including the tie window after each pick -- is the oracle's. */
__device__ __forceinline__ bool r3d_after_keep(float v, int64_t i, float pv, int64_t pi) {
    if (v > pv) return false;              /* strictly better than the last pick */
    if (v == pv && i <= pi) return false;  /* the last pick itself, or one before it */
    return true;
}

__global__ __launch_bounds__(kThreads) void row_topk_kernel(
    const uint8_t* __restrict__ x, uint32_t dt, int64_t x_pitch,
    int32_t* __restrict__ idx, int64_t idx_pitch,
    float* __restrict__ val, int64_t val_pitch,
    float* __restrict__ pairs, int64_t pairs_pitch,   /* byte pitch, positioned then cast */
    int64_t N, int64_t R, int64_t off) {
    __shared__ float s_v[kWarps]; __shared__ int64_t s_i[kWarps];
    __shared__ float s_pv; __shared__ int64_t s_pi;
    const int64_t m = blockIdx.x;
    const uint8_t* __restrict__ xr = x + m * x_pitch;
    if (threadIdx.x == 0) { s_pv = INFINITY; s_pi = -1; }
    __syncthreads();

    for (int64_t r = 0; r < R; ++r) {
        const float pv = s_pv; const int64_t pi = s_pi;
        float bv = 0.0f; int64_t bi = -1;
        for (int64_t i = (int64_t)threadIdx.x; i < N; i += (int64_t)kThreads) {
            const float v = ldt2(xr, dt, i);
            if (r3d_after_keep(v, i, pv, pi) && r3d_beats(v, i, bv, bi)) { bv = v; bi = i; }
        }
        for (uint32_t sh = 16; sh; sh >>= 1) {
            const float sv = __shfl_down_sync(0xFFFFFFFFull, bv, sh);
            const int64_t si = __shfl_down_sync(0xFFFFFFFFull, bi, sh);
            if (r3d_beats(sv, si, bv, bi)) { bv = sv; bi = si; }
        }
        const uint32_t lane = threadIdx.x & 31u, wid = threadIdx.x >> 5;
        if (lane == 0) { s_v[wid] = bv; s_i[wid] = bi; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float rv = s_v[0]; int64_t ri = s_i[0];
            for (uint32_t w = 1; w < kWarps; ++w)
                if (r3d_beats(s_v[w], s_i[w], rv, ri)) { rv = s_v[w]; ri = s_i[w]; }
            const bool ok = ri >= 0;
            idx[m * idx_pitch + r] = ok ? (int32_t)(ri + off) : -1;
            val[m * val_pitch + r] = ok ? rv : -INFINITY;
            if (pairs) {
                float* const p = (float*)((uint8_t*)pairs + m * pairs_pitch) + r * 2;
                const int32_t i32 = ok ? (int32_t)(ri + off) : -1;
                uint32_t bits; memcpy(&bits, &i32, 4);
                p[0] = __uint_as_float(bits);
                p[1] = ok ? rv : -INFINITY;
            }
            /* An exhausted row keeps its (pv, pi) -- ref breaks out and writes the
             * -1/-inf holes for every remaining r, which the code above does each
             * round once ok is false. */
            if (ok) { s_pv = rv; s_pi = ri; }
        }
        __syncthreads();
    }
}

/* Shared host-side helpers (pattern-copied from r3d_band.hip). */
static inline bool dt_act_ok(uint32_t dt) { return dt == RAD_F32 || dt == RAD_BF16; }
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
extern "C" int r3d_rad_silu_mul(const RadArgs* a, RadStream s) {
    const RadTensor* gu = rad_arg_in(a, 0);
    const RadTensor* y  = rad_arg_in(a, 1);
    if (!gu || !y) return RAD_E_INVAL;
    if (!dt_act_ok(gu->dtype) || y->dtype != gu->dtype) return RAD_E_DTYPE;
    const int64_t n = param_i(a, "n", y->rank >= 2 ? y->shape[1] : 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rad_tensor_numel(y) / n;
    if (rows <= 0 || rows > 0xffff) return RAD_E_SHAPE;
    if (rad_tensor_numel(gu) < rows * 2 * n) return RAD_E_SHAPE;
    int64_t gp, yp;
    if (row2_pitch(gu, 2 * n, &gp)) return RAD_E_STRIDE;
    if (row2_pitch(y, n, &yp)) return RAD_E_STRIDE;
    const int64_t total = rows * n;
    uint32_t blocks = (uint32_t)((total + kThreads - 1) / kThreads);
    if (blocks > 0xffff) blocks = 0xffff;
    silu_mul_kernel<<<dim3(blocks), dim3(kThreads), 0, (hipStream_t)(uintptr_t)s>>>(
        (const uint8_t*)gu->data, gu->dtype, rad_dtype_bytes(gu->dtype, gp),
        (uint8_t*)y->data, rad_dtype_bytes(y->dtype, yp), n, total);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r3d_rad_quant_act_i8(const RadArgs* a, RadStream s) {
    const RadTensor* x = rad_arg_in(a, 0);
    const RadTensor* q = rad_arg_in(a, 1);
    const RadTensor* sc = rad_arg_in(a, 2);
    const RadTensor* as = (a->n_t > 3 && a->t[3].data) ? &a->t[3] : nullptr;
    if (!x || !q || !sc) return RAD_E_INVAL;
    if (!dt_act_ok(x->dtype)) return RAD_E_DTYPE;
    if (q->dtype != RAD_I8 || sc->dtype != RAD_F32) return RAD_E_DTYPE;
    const int64_t n = param_i(a, "n", 0);
    if (n <= 0 || n % 16) return RAD_E_SHAPE;             /* whole steps */
    if (q->rank != 2 || q->stride[1] != 1) return RAD_E_STRIDE;
    const int64_t rows = rad_tensor_numel(x) / n;
    if (rows <= 0 || rows > 0xffff || rad_tensor_numel(q) < rows * n
        || rad_tensor_numel(sc) < rows) return RAD_E_SHAPE;
    const int64_t g = as ? param_i(a, "group", 0) : 0;
    const int64_t mpad = (rows + 15) / 16 * 16;
    if (as && (g <= 0 || n % g || rad_tensor_numel(as) < n / g * mpad)) return RAD_E_SHAPE;
    int64_t xp, qp, ap = 0;
    if (row2_pitch(x, n, &xp)) return RAD_E_STRIDE;
    if (q->rank == 1) {                       /* a single row may come flat */
        if (q->stride[0] != 1 || rad_tensor_numel(q) < n) return RAD_E_STRIDE;
        qp = n;                               /* row 0 only: the pitch is unused */
    } else if (q->rank == 2 && q->stride[1] == 1 && q->stride[0] >= n) {
        qp = q->stride[0];
    } else return RAD_E_STRIDE;
    /* scale is [M] or [M, 1] -- contiguous either way. */
    if (sc->rank > 2 || (sc->rank == 2 && (sc->stride[1] != 1 || sc->stride[0] != 1)))
        return RAD_E_STRIDE;
    if (as) {
        if (as->dtype != RAD_F32 || as->rank != 2 || as->stride[1] != 1
            || as->shape[0] < n / g || as->shape[1] < mpad) return RAD_E_SHAPE;
        ap = as->stride[0];
    }
    quant_act_i8_kernel<<<dim3((uint32_t)rows), dim3(kThreads), 0,
                          (hipStream_t)(uintptr_t)s>>>(
        (const uint8_t*)x->data, x->dtype, rad_dtype_bytes(x->dtype, xp),
        (int8_t*)q->data, qp, (float*)sc->data,
        as ? (float*)as->data : nullptr, ap, n, g);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r3d_rad_row_topk(const RadArgs* a, RadStream s) {
    const RadTensor* x = rad_arg_in(a, 0);
    const RadTensor* idx = rad_arg_in(a, 1);
    const RadTensor* val = rad_arg_in(a, 2);
    const RadTensor* pairs = (a->n_t > 3 && a->t[3].data) ? &a->t[3] : nullptr;
    if (!x || !idx || !val) return RAD_E_INVAL;
    if (!dt_act_ok(x->dtype)) return RAD_E_DTYPE;
    if (idx->dtype != RAD_I32 || val->dtype != RAD_F32) return RAD_E_DTYPE;
    if (pairs && pairs->dtype != RAD_F32) return RAD_E_DTYPE;
    const int64_t N = param_i(a, "N", x->rank >= 2 ? x->shape[1] : 0);
    const int64_t R = param_i(a, "R", 1);
    const int64_t off = param_i(a, "vocab_off", 0);
    if (N <= 0 || R <= 0 || R > 32) return RAD_E_SHAPE;
    const int64_t M = rad_tensor_numel(x) / N;
    if (M <= 0 || M > 0xffff || rad_tensor_numel(idx) < M * R
        || rad_tensor_numel(val) < M * R) return RAD_E_SHAPE;
    if (pairs && rad_tensor_numel(pairs) < M * R * 2) return RAD_E_SHAPE;
    int64_t xp, ip, vp, pp = 0;
    if (row2_pitch(x, N, &xp)) return RAD_E_STRIDE;
    if (row2_pitch(idx, R, &ip)) return RAD_E_STRIDE;
    if (row2_pitch(val, R, &vp)) return RAD_E_STRIDE;
    /* The fixture's pairs plane is rank 3 [M, R, 2] (the shape the schema describes); a
     * flat rank-2 [M, 2R] is the same memory and is accepted too. Any other stride
     * layout is refused rather than read at the wrong offsets. */
    if (pairs) {
        if (pairs->rank == 3) {
            if (pairs->stride[2] != 1 || pairs->stride[1] != 2 ||
                pairs->stride[0] != 2 * R) return RAD_E_STRIDE;
            pp = 2 * R;
        } else if (row2_pitch(pairs, 2 * R, &pp)) return RAD_E_STRIDE;
    }
    row_topk_kernel<<<dim3((uint32_t)M), dim3(kThreads), 0, (hipStream_t)(uintptr_t)s>>>(
        (const uint8_t*)x->data, x->dtype, rad_dtype_bytes(x->dtype, xp),
        (int32_t*)idx->data, ip, (float*)val->data, vp,
        pairs ? (float*)pairs->data : nullptr, pairs ? rad_dtype_bytes(pairs->dtype, pp) : 0,
        N, R, off);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
