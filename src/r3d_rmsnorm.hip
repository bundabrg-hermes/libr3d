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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include "r3d_dump.h"

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
/* Q32: the store chain for the two bf16s packed in `pair`, per-element order
 * verbatim from the scalar kernel below: (x * s) * (g + wadd) * mult, one RTNE
 * narrowing per element. A pure function of the inputs -- same expression, same
 * f32 associativity, so the two kernels agree element-for-element given the same s. */
__device__ __forceinline__ uint32_t r3d_scale_pair(uint32_t pair, float s,
                                                   float g0, float g1,
                                                   float wadd, float mult) {
    const float y0 = r3d_bf16_to_f32((uint16_t)pair)         * s * (g0 + wadd) * mult;
    const float y1 = r3d_bf16_to_f32((uint16_t)(pair >> 16)) * s * (g1 + wadd) * mult;
    return (uint32_t)r3d_f32_to_bf16(y0) | ((uint32_t)r3d_f32_to_bf16(y1) << 16);
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

/* Q32: the 128-bit vectorisation the header earmarked at :17 -- 8 contiguous bf16
 * per thread as ONE uint4 (gfx11's global_load/store_b128), for rows the entry
 * gates in below (n >= 2048, n % 8 == 0, 16B-aligned base AND row pitch % 8 == 0).
 * The per-element store chain is the scalar kernel's verbatim (same f32 op order,
 * same single RTNE narrowing, -ffp-contract=off), so the ONLY arithmetic delta is
 * the sum-of-squares reassociation: thread ownership is 8-wide chunks instead of
 * stride-kThreads elements. Head dims (128/256), ragged and misaligned rows stay
 * on the untouched scalar kernel above; nothing here can perturb them. */
template <bool WF32>
__global__ __launch_bounds__(kThreads) void rmsnorm_kernel_vec(
    const uint16_t* __restrict__ x, const void* __restrict__ w, uint16_t* __restrict__ y,
    uint32_t n, float eps, float wadd, float mult, uint32_t x_ld, uint32_t y_ld) {
    __shared__ float lds[kWaves];
    const uint32_t m = blockIdx.x;
    const uint4* __restrict__ xr = (const uint4*)(x + (size_t)m * x_ld);
    uint4* __restrict__ yr = (uint4*)(y + (size_t)m * y_ld);
    const uint32_t nv = n / 8;

    float ss = 0.0f;
    for (uint32_t v = threadIdx.x; v < nv; v += kThreads) {
        const uint4 u = xr[v];
        const float a = r3d_bf16_to_f32((uint16_t)u.x), b = r3d_bf16_to_f32((uint16_t)(u.x >> 16));
        const float c = r3d_bf16_to_f32((uint16_t)u.y), d = r3d_bf16_to_f32((uint16_t)(u.y >> 16));
        const float e = r3d_bf16_to_f32((uint16_t)u.z), f = r3d_bf16_to_f32((uint16_t)(u.z >> 16));
        const float g = r3d_bf16_to_f32((uint16_t)u.w), h = r3d_bf16_to_f32((uint16_t)(u.w >> 16));
        ss += ((a*a + b*b) + (c*c + d*d)) + ((e*e + f*f) + (g*g + h*h));
    }
    ss = r3d_block_sum(ss, lds);
    const float s = rsqrtf(ss / (float)n + eps);

    for (uint32_t v = threadIdx.x; v < nv; v += kThreads) {
        const uint4 u = xr[v];
        uint4 o;
        if (!w) {
            o.x = r3d_scale_pair(u.x, s, 1.0f, 1.0f, wadd, mult);   /* g == 1 verbatim: */
            o.y = r3d_scale_pair(u.y, s, 1.0f, 1.0f, wadd, mult);   /* the scalar path  */
            o.z = r3d_scale_pair(u.z, s, 1.0f, 1.0f, wadd, mult);   /* adds (1 + wadd)  */
            o.w = r3d_scale_pair(u.w, s, 1.0f, 1.0f, wadd, mult);   /* as (g + wadd)   */
        } else if (WF32) {
            const float4* __restrict__ wr = (const float4*)((const float*)w + (size_t)v * 8);
            const float4 wa = wr[0], wb = wr[1];
            o.x = r3d_scale_pair(u.x, s, wa.x, wa.y, wadd, mult);
            o.y = r3d_scale_pair(u.y, s, wa.z, wa.w, wadd, mult);
            o.z = r3d_scale_pair(u.z, s, wb.x, wb.y, wadd, mult);
            o.w = r3d_scale_pair(u.w, s, wb.z, wb.w, wadd, mult);
        } else {
            const uint4 wv = ((const uint4*)w)[v];
            o.x = r3d_scale_pair(u.x, s, r3d_bf16_to_f32((uint16_t)wv.x),
                                 r3d_bf16_to_f32((uint16_t)(wv.x >> 16)), wadd, mult);
            o.y = r3d_scale_pair(u.y, s, r3d_bf16_to_f32((uint16_t)wv.y),
                                 r3d_bf16_to_f32((uint16_t)(wv.y >> 16)), wadd, mult);
            o.z = r3d_scale_pair(u.z, s, r3d_bf16_to_f32((uint16_t)wv.z),
                                 r3d_bf16_to_f32((uint16_t)(wv.z >> 16)), wadd, mult);
            o.w = r3d_scale_pair(u.w, s, r3d_bf16_to_f32((uint16_t)wv.w),
                                 r3d_bf16_to_f32((uint16_t)(wv.w >> 16)), wadd, mult);
        }
        yr[v] = o;
    }
}

} /* anonymous namespace */

/* The launch entry, named by the row table. Operands are positional in libref's rmsnorm schema
 * order: x, w (optional), y. A launch must not allocate and must not synchronise. */
static int r3d_rmsnorm_impl(const RadArgs*, RadStream);
extern "C" int r3d_rad_rmsnorm(const RadArgs* a, RadStream s)
    R3D_DUMP_WRAP("rmsnorm_bf16", r3d_rmsnorm_impl)
static int r3d_rmsnorm_impl(const RadArgs* a, RadStream s) {
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

    /* Q32 lever. PROMOTED DEFAULT-ON after adjudication (A/B: +54-56% at every
     * gated shape, hot and cold, disjoint ranges). R3D_RMS_VEC=0 is the
     * kill-switch back to the scalar kernel. The vector arm takes only rows the
     * b128 walk can serve unchanged: n >= 2048 (head dims 128/256 and every
     * small row stay scalar by geometry), n % 8 == 0, both row pitches a
     * multiple of 8 (every row base then inherits the base alignment), and
     * 16B-aligned x/y/w bases. Any miss falls to the untouched scalar kernel --
     * refuse, never approximate. */
    static const int rms_vec = []{
        const char* e = getenv("R3D_RMS_VEC");
        return !e || atoi(e) != 0 ? 1 : 0;
    }();
    const bool vec_ok = rms_vec && n >= 2048 && (n % 8) == 0
                     && (x_ld % 8u) == 0 && (y_ld % 8u) == 0
                     && (((uintptr_t)x->data & 15u) | ((uintptr_t)y->data & 15u)
                         | (w ? (uintptr_t)w->data & 15u : 0u)) == 0;
    if (getenv("R3D_DEBUG_RMS"))
        fprintf(stderr, "[r3d] rmsnorm arm=%s n=%u M=%lld w=%d\n",
                vec_ok ? "vec" : "scalar", (uint32_t)n, (long long)M, w ? 1 : 0);

    if (w && w->dtype == RAD_F32) {
        if (vec_ok)
            hipLaunchKernelGGL((rmsnorm_kernel_vec<true>), dim3((uint32_t)M), dim3(kThreads), 0, stream,
                               (const uint16_t*)x->data, w->data, (uint16_t*)y->data,
                               (uint32_t)n, (float)eps, (float)wadd, (float)mult, x_ld, y_ld);
        else
            hipLaunchKernelGGL((rmsnorm_kernel<true>), dim3((uint32_t)M), dim3(kThreads), 0, stream,
                               (const uint16_t*)x->data, w->data, (uint16_t*)y->data,
                               (uint32_t)n, (float)eps, (float)wadd, (float)mult, x_ld, y_ld);
    } else {
        if (vec_ok)
            hipLaunchKernelGGL((rmsnorm_kernel_vec<false>), dim3((uint32_t)M), dim3(kThreads), 0, stream,
                               (const uint16_t*)x->data, w ? w->data : nullptr, (uint16_t*)y->data,
                               (uint32_t)n, (float)eps, (float)wadd, (float)mult, x_ld, y_ld);
        else
            hipLaunchKernelGGL((rmsnorm_kernel<false>), dim3((uint32_t)M), dim3(kThreads), 0, stream,
                               (const uint16_t*)x->data, w ? w->data : nullptr, (uint16_t*)y->data,
                               (uint32_t)n, (float)eps, (float)wadd, (float)mult, x_ld, y_ld);
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
