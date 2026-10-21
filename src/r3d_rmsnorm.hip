/* r3d_rmsnorm.hip -- libr3d's first kernel: the `rmsnorm` op on RDNA3.
 *
 * The arithmetic is libref's (ref_elementwise.cpp rms_rows, Apache-2.0, (c) radiance authors --
 * ported with attribution), with the rounding in the same places:
 *   * the sum of squares accumulates in f32 and only the store narrows to bf16;
 *   * the store is round-to-nearest-even (r3d_f32_to_bf16 below);
 *   * `-ffp-contract=off` (set in CMakeLists): the oracle forbids FMA contraction;
 *   * `wadd` (GemmaRMSNorm's `1 +`) is added in f32 after the load, never pre-formed in bf16;
 *   * `mult` scales the normed row before its single rounding.
 *
 * Unlike ref's per-thread ascending sum, each thread strides the row by the block size and the
 * block tree-reduces: an f32 reassociation, tolerated by kbench's tolerance the same way libr4d's
 * identical structure is (PLUGIN.md: reductions reassociate, stores do not).
 *
 * Nothing here is RDNA4- or RDNA3-specific -- plain HIP, one wave of __shfl_down_sync (gfx11's
 * __reduce_add_sync has no f32 form; gfx12's does, so staying shfl-based keeps one source for
 * both). The headroom for later passes is 128-bit row loads, which need the same alignment check
 * libr4d does via rad_ptr_aligned.
 */

#include <hip/hip_runtime.h>

#include "rad_abi.h"
#include "rad_plugin.h"

/* The wave32 ABI (which this file's block reduce assumes, counting 32-lane waves) is pinned in
 * CMakeLists with -mno-wavefrontsize64, not asserted here: ROCm 10.1's clang defines no
 * wave-size macro and its __builtin_amdgcn_wave_size() is not a constant expression. */

namespace {

constexpr uint32_t kWave    = 32;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kWaves   = kThreads / kWave;

/* Device-side narrowings. rad_plugin.h's identical conversions are host inline functions;
 * these mirror them bit for bit: bf16 widening is the 16-bit left shift, narrowing is RTNE
 * with the carry-into-infinity and quiet-NaN cases spelled out there. */
__device__ __forceinline__ float r3d_bf16_to_f32(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}
__device__ __forceinline__ uint16_t r3d_f32_to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ float r3d_wave_sum(float v) {
    /* ROCm 10.1's masked shuffle wants a 64-bit mask (static_assert in
     * amd_warp_sync_functions.h); the low 32 bits are the full wave under the wave32 ABI. */
    const uint64_t full = ~0ull >> 32;
#pragma unroll
    for (uint32_t off = kWave / 2; off; off >>= 1)
        v += __shfl_down_sync(full, v, off);
    return v;
}

/* Sum across the block, broadcast: one LDS round trip after the wave reduction, as in libr4d. */
__device__ __forceinline__ float r3d_block_sum(float v, float* lds) {
    v = r3d_wave_sum(v);
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < kWaves; ++w) s += lds[w];
    return s;
}

/* ref_rmsnorm: y = x * rsqrt(mean(x^2) + eps) * (w + wadd) * mult.
 *
 * One block per row; `WF32` selects an f32 gain (architecture plugins declare norm gains f32,
 * checkpoints carry bf16). Two passes over the row: n is a head's 128/256 or the hidden state's
 * ~2-6k, so the row is in cache for the second pass. Absent gain is passed as w == nullptr. */
template <bool WF32>
__global__ __launch_bounds__(kThreads) void rmsnorm_kernel(
    const uint16_t* __restrict__ x, const void* __restrict__ w, uint16_t* __restrict__ y,
    uint32_t n, float eps, float wadd, float mult, uint32_t x_ld, uint32_t y_ld) {
    __shared__ float lds[kWaves];
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;

    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
        const float v = r3d_bf16_to_f32(xr[i]);
        ss += v * v;
    }
    ss = r3d_block_sum(ss, lds);
    const float s = rsqrtf(ss / (float)n + eps);

    for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
        const float g = !w ? 1.0f
                      : WF32 ? ((const float*)w)[i]
                             : r3d_bf16_to_f32(((const uint16_t*)w)[i]);
        yr[i] = r3d_f32_to_bf16(r3d_bf16_to_f32(xr[i]) * s * (g + wadd) * mult);
    }
}

} /* anonymous namespace */

/* The launch entry, named by the row table. Operands are positional in libref's rmsnorm schema
 * order: x, w (optional), y. A launch must not allocate and must not synchronise. */
extern "C" int r3d_rad_rmsnorm(const RadArgs* a, RadStream s) {
    const RadTensor* x = rad_arg_in(a, 0);
    const RadTensor* w = rad_arg_in(a, 1);
    const RadTensor* y = rad_arg_in(a, 2);
    if (!x || !y) return RAD_E_INVAL;
    /* v1 serves the bf16 activation row. The row table's `dtype` constraint keeps the core from
     * selecting this row otherwise; this check is the cold-call guard (kbench, selftest). */
    if (x->dtype != RAD_BF16 || y->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (w && w->dtype != RAD_F32 && w->dtype != RAD_BF16) return RAD_E_DTYPE;

    /* n: the declared parameter, else the gain's width, else the row extent (ref's rule). */
    long long n = 0;
    if (!rad_args_geti(a, "n", &n) || n <= 0)
        n = w ? rad_tensor_lastdim(w) : rad_tensor_lastdim(y);
    if (n <= 0 || n > 0x7fffffffLL) return RAD_E_SHAPE;

    /* Ref's rowoff: a last extent of n reads the row stride; a wider last extent (a flat
     * [T, m*n] viewed as [T*m, n]) reads the flat pitch n, unit in-row stride. A rank-3+
     * with last extent n is served only when the leading dims are contiguous, so the flat
     * walk r*n lands where ref's per-dim walk lands. */
    if (rad_tensor_lastdim(x) != n && rad_tensor_lastdim(x) % n != 0) return RAD_E_SHAPE;
    if (rad_tensor_lastdim(y) != n && rad_tensor_lastdim(y) % n != 0) return RAD_E_SHAPE;
    const int64_t x_ls = x->rank ? x->stride[x->rank - 1] : 0;
    const int64_t y_ls = y->rank ? y->stride[y->rank - 1] : 0;
    if (x_ls != 1 || y_ls != 1) return RAD_E_STRIDE;
    if (w && (w->rank != 1 || w->stride[0] != 1 || rad_tensor_lastdim(w) < n))
        return RAD_E_STRIDE;
    auto ld_row_pitch = [&](const RadTensor* t) -> uint32_t {
        if (t->rank <= 1) return (uint32_t)n;
        const int64_t ld = t->shape[t->rank - 1];
        if (ld != n) return (uint32_t)n;                      /* flat view: pitch = n */
        if (t->rank >= 3) {
            for (int i = 0; i + 1 < (int)t->rank - 1; ++i)    /* walk == flat only contiguous */
                if (t->stride[i] != t->shape[i + 1] * t->stride[i + 1]) return 0;
        }
        return (uint32_t)t->stride[t->rank - 2];
    };

    /* Rows count numel/n, exactly as ref: a flat wide view is cut into n-wide rows. */
    const int64_t M = rad_tensor_numel(x) / n;
    if (M <= 0 || rad_tensor_numel(y) < M * n) return RAD_E_SHAPE;

    const double eps  = rad_args_getf_or(a, "eps", 1e-6);
    const double wadd = rad_args_getf_or(a, "wadd", 0.0);
    const double mult = rad_args_getf_or(a, "mult", 1.0);
    /* ref refuses `wadd` without a gain; so does this row, for the same reason. */
    if (!w && wadd != 0.0) return RAD_E_INVAL;

    const uint32_t x_ld = ld_row_pitch(x);
    const uint32_t y_ld = ld_row_pitch(y);
    if (!x_ld || !y_ld) return RAD_E_STRIDE;     /* rank-3+ with non-contiguous leading dims */
    if ((int64_t)(M - 1) + n > (int64_t)0x7fffffffLL) return RAD_E_SHAPE;

    hipStream_t stream = (hipStream_t)(uintptr_t)s;
    if (w && w->dtype == RAD_F32)
        hipLaunchKernelGGL((rmsnorm_kernel<true>), dim3((uint32_t)M), dim3(kThreads), 0, stream,
                           (const uint16_t*)x->data, w->data, (uint16_t*)y->data,
                           (uint32_t)n, (float)eps, (float)wadd, (float)mult, x_ld, y_ld);
    else
        hipLaunchKernelGGL((rmsnorm_kernel<false>), dim3((uint32_t)M), dim3(kThreads), 0, stream,
                           (const uint16_t*)x->data, w ? w->data : nullptr, (uint16_t*)y->data,
                           (uint32_t)n, (float)eps, (float)wadd, (float)mult, x_ld, y_ld);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
