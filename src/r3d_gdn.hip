/* r3d_gdn.hip -- libr3d's first steps into the Gated DeltaNet family, the op set 48 of
 * the qwen3.5-27B trunk's 64 layers run on. The family opener is the cheapest member:
 * `gdn_gated_rmsnorm`, the epilogue norm the recurrent update also folds into itself.
 * The rest of the family (conv_update, kkt_solve, chunk_scan, recurrent_update) joins
 * this file; the arithmetic stays libref's (ref_gdn.cpp, Apache-2.0, (c) radiance
 * authors -- ported with attribution).
 *
 * gdn_gated_rmsnorm: o = x * rsqrt(mean(x^2) + eps) * w * act(z), act in {silu, sigmoid},
 * one row per (token, value head):
 *   * the row sum is a block reduction in f32; ref accumulates it sequentially, and the
 *     tree-vs-sequence delta lives at 1e-7 on the row scale -- the same rounding class
 *     every reduced device row of this library already trades (r3d_rmsnorm);
 *   * the scaling keeps ref's exact op order: sc = 1/sqrt(ss/ch + eps), then the store
 *     chain ((x*sc)*w)*gate, each step a separate f32 (-ffp-contract=off) and the bf16
 *     narrowing happens once, at the store;
 *   * the gate reuses the band's double-rounded sigmoid (r3d_sigmoid), silu = z*sigmoid(z);
 *   * `z` may be a column slice of a wider fused row -- ref's gate_rowoff -- which the
 *     kernel walks as (row/heads)*z_pitch + (row%heads)*ch.
 */
#include <cstring>

#include <hip/hip_runtime.h>

#include "rad_abi.h"
#include "rad_plugin.h"

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kWarps   = kThreads / 32;

__device__ __forceinline__ float r3d_exp_f(float x) { return (float)exp((double)x); }
__device__ __forceinline__ float r3d_sigmoid(float x) { return 1.0f / (1.0f + r3d_exp_f(-x)); }

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
__device__ __forceinline__ void stt2(uint8_t* p, uint32_t dt, int64_t i, float v) {
    if (dt == RAD_F32) ((float*)p)[i] = v;
    else ((uint16_t*)p)[i] = r3d_f32_to_bf16(v);
}

/* Block sum of per-thread partials; order is tree by construction, see the file header
 * for why that is the same rounding class the shipped rmsnorm row already lives in. */
__device__ __forceinline__ float block_sum(float v, float* s_part) {
    for (uint32_t off = 16; off; off >>= 1)
        v += __shfl_down_sync(0xFFFFFFFFull, v, off);
    const uint32_t lane = threadIdx.x & 31u, wid = threadIdx.x >> 5;
    if (lane == 0) s_part[wid] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float r = s_part[0];
        for (uint32_t w = 1; w < kWarps; ++w) r += s_part[w];
        s_part[0] = r;
    }
    __syncthreads();
    return s_part[0];
}

/* One block per row. Pitches are in BYTES (rad_dtype_bytes applied at the entry); the
 * element strides within a row are 1 -- wider unit-strides are refused there, as every
 * other row of this library refuses them. z_heads > 1 selects the gate_rowoff slice
 * walk: z row = (r/z_heads)*z_tok + (r%z_heads)*ch elements. */
__global__ __launch_bounds__(kThreads) void gdn_gated_rmsnorm_kernel(
    const uint8_t* __restrict__ x, uint32_t dtX, int64_t x_pitch,
    const uint8_t* __restrict__ z, uint32_t dtZ, int64_t z_pitch,
    int64_t z_tok, int32_t z_heads,
    const uint8_t* __restrict__ w, uint32_t dtW,
    uint8_t* __restrict__ o, uint32_t dtO, int64_t o_pitch,
    int64_t ch, float eps, int sigm) {
    __shared__ float s_part[kWarps]; __shared__ float s_sc;
    const int64_t r = blockIdx.x;
    const uint8_t* __restrict__ xb = x + r * x_pitch;
    uint8_t* __restrict__ ob = o + r * o_pitch;
    const int64_t zb_elems = (z_heads > 1)
        ? (r / z_heads) * z_tok + (r % z_heads) * ch
        : r * (z_pitch / (dtZ == RAD_F32 ? 4 : 2));

    float ss = 0.0f;
    for (int64_t i = (int64_t)threadIdx.x; i < ch; i += (int64_t)kThreads) {
        const float u = ldt2(xb, dtX, i);
        ss += u * u;
    }
    ss = block_sum(ss, s_part);
    if (threadIdx.x == 0) s_sc = 1.0f / sqrtf(ss / (float)ch + eps);
    __syncthreads();
    const float sc = s_sc;

    const uint8_t* __restrict__ zb = z + zb_elems * (dtZ == RAD_F32 ? 4 : 2);
    for (int64_t i = (int64_t)threadIdx.x; i < ch; i += (int64_t)kThreads) {
        const float zz = ldt2(zb, dtZ, i);
        const float gate = sigm ? r3d_sigmoid(zz) : zz * r3d_sigmoid(zz);
        const float v = ((ldt2(xb, dtX, i) * sc) * ldt2(w, dtW, i)) * gate;
        stt2(ob, dtO, i, v);
    }
}

/* Ref's rowoff for the dense class this row serves: rank-2 [rows, cols] with a unit
 * last stride (pitch = stride[0]), or a bare rank-1 single row (pitch = cols). */
static inline int row2_pitch(const RadTensor* t, int64_t cols, int64_t* pitch) {
    if (!t || !t->data) return -1;
    if (t->rank == 1) {
        if (rad_tensor_numel(t) != cols || t->stride[0] != 1) return -2;
        *pitch = cols;
        return 0;
    }
    if (t->rank != 2 || t->shape[1] != cols || t->stride[1] != 1) return -3;
    *pitch = t->stride[0];
    return 0;
}

} /* anonymous namespace */

extern "C" int r3d_rad_gdn_gated_rmsnorm(const RadArgs* a, RadStream s) {
    const RadTensor* x = rad_arg_in(a, 0);
    const RadTensor* z = rad_arg_in(a, 1);
    const RadTensor* w = rad_arg_in(a, 2);
    const RadTensor* o = rad_arg_in(a, 3);
    if (!x || !z || !w || !o) return RAD_E_INVAL;
    const bool dt_ok = x->dtype == RAD_F32 || x->dtype == RAD_BF16;
    const bool dt_z = z->dtype == RAD_F32 || z->dtype == RAD_BF16;
    const bool dt_w = w->dtype == RAD_F32 || w->dtype == RAD_BF16;
    const bool dt_o = o->dtype == RAD_F32 || o->dtype == RAD_BF16;
    if (!dt_ok || !dt_z || !dt_w || !dt_o) return RAD_E_DTYPE;

    long long ch_ll = 0;
    if (!rad_args_geti(a, "channels", &ch_ll))
        ch_ll = w->rank ? w->shape[w->rank - 1] : 0;
    const int64_t ch = ch_ll;
    if (ch <= 0 || ch > 0xffff) return RAD_E_SHAPE;

    const double eps_d = rad_args_getf_or(a, "eps", 1e-6);
    const float eps = (float)eps_d;

    const char* act = rad_args_gets(a, "act");
    const int sigm = act && !std::strcmp(act, "sigmoid");
    if (act && !sigm && std::strcmp(act, "silu")) return RAD_E_UNSUPPORTED;

    const int64_t rows = rad_tensor_numel(x) / ch;
    if (rows <= 0 || rows > 0xffff) return RAD_E_SHAPE;
    if (rad_tensor_numel(o) < rows * ch || rad_tensor_numel(z) < rows * ch) return RAD_E_SHAPE;
    if (rad_tensor_numel(w) < ch) return RAD_E_SHAPE;
    if (w->rank != 1 || w->stride[0] != 1) return RAD_E_STRIDE;

    int64_t xp, op;
    if (row2_pitch(x, ch, &xp)) return RAD_E_STRIDE;
    if (row2_pitch(o, ch, &op)) return RAD_E_STRIDE;

    /* The gate: a plain [rows, ch] row, or ref's gate_rowoff slice -- rank-2 whose row
     * is heads*ch wide, read heads-at-a-time. A rank-1 z is the single-row case. */
    int64_t z_pitch = 0, z_tok = 0; int32_t z_heads = 1;
    if (z->rank == 2 && z->shape[1] != ch && z->shape[1] % ch == 0 && z->stride[1] == 1) {
        z_heads = (int32_t)(z->shape[1] / ch);
        z_tok = z->stride[0];
    } else if (row2_pitch(z, ch, &z_pitch)) {
        return RAD_E_STRIDE;
    }

    gdn_gated_rmsnorm_kernel<<<dim3((uint32_t)rows), dim3(kThreads), 0,
                               (hipStream_t)(uintptr_t)s>>>(
        (const uint8_t*)x->data, x->dtype, rad_dtype_bytes(x->dtype, xp),
        (const uint8_t*)z->data, z->dtype, rad_dtype_bytes(z->dtype, z_pitch),
        z_tok, z_heads,
        (const uint8_t*)w->data, w->dtype,
        (uint8_t*)o->data, o->dtype, rad_dtype_bytes(o->dtype, op),
        ch, eps, sigm);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
