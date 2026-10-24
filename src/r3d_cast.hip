/* r3d_cast.hip -- libr3d's `cast` on RDNA3: the dtype edges.

 * The semantics are libref's (ref_elementwise.cpp ref_cast, Apache-2.0, (c) radiance authors):
 * a linear walk over numel(y), x widened or narrowed element by element through f32. Three
 * pairs share this entry, one row each:
 *   * bf16 -> bf16: a pure copy (a strided row of a wider buffer into another);
 *   * bf16 -> f32: the exact widening (a bf16 is a truncated f32; no rounding exists);
 *   * f32 -> bf16: one RTNE narrowing, the same helper the other libr3d kernels store through.
 * Pairs outside those three (the f16 narrowing, any integer edge) are refused at launch; the
 * rows' `from`/`to` constraints keep the selector from asking. The schema notes the params are
 * for the selector and the tensors' own dtypes are what the kernel reads -- the entry follows
 * the tensors.
 *
 * Both operands keep their own row pitch: M and n are the declared extent, and a tensor sliced
 * out of a wider allocation (pitch > n) is exactly what a column tap produces.
 */

#include <hip/hip_runtime.h>
#include "r3d_dump.h"

#include "rad_abi.h"
#include "rad_plugin.h"

namespace {

constexpr uint32_t kThreads = 256;

__device__ __forceinline__ float r3d_bf16_to_f32(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}
__device__ __forceinline__ uint16_t r3d_f32_to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

template <int Mode, int Contig>   /* Mode: 0 copy, 1 widen bf16->f32, 2 narrow f32->bf16.
    * Contig: x_pitch == y_pitch == n, so r*pitch+i == e for both faces and the per-element
    * 64-bit division is dead at compile time -- the contiguous walk is a pure linear copy.
    * Both arms address the identical element per e, so outputs are bit-identical. */
__global__ __launch_bounds__(kThreads) void cast_kernel(
    const void* __restrict__ x, void* __restrict__ y, uint64_t total, uint32_t n,
    int64_t x_pitch, int64_t y_pitch) {
    const uint64_t idx = (uint64_t)blockIdx.x * kThreads + threadIdx.x;
    for (uint64_t e = idx; e < total; e += (uint64_t)gridDim.x * kThreads) {
        const uint64_t r = Contig ? 0 : e / n;
        const uint64_t ox = Contig ? e : r * (uint64_t)x_pitch + (e - r * n);
        const uint64_t oy = Contig ? e : r * (uint64_t)y_pitch + (e - r * n);
        if (Mode == 0)
            ((uint16_t*)y)[oy] = ((const uint16_t*)x)[ox];
        else if (Mode == 1)
            ((float*)y)[oy] = r3d_bf16_to_f32(((const uint16_t*)x)[ox]);
        else
            ((uint16_t*)y)[oy] = r3d_f32_to_bf16(((const float*)x)[ox]);
    }
}

} /* anonymous namespace */

/* Launch entry, operands positional in the cast schema: x, y. */
static int r3d_cast_impl(const RadArgs*, RadStream);
extern "C" int r3d_rad_cast(const RadArgs* a, RadStream s)
    R3D_DUMP_WRAP("cast", r3d_cast_impl)
static int r3d_cast_impl(const RadArgs* a, RadStream s) {
    const RadTensor* X = rad_arg_in(a, 0);
    const RadTensor* Y = rad_arg_in(a, 1);
    if (!X || !Y) return RAD_E_INVAL;

    int mode;
    if (X->dtype == RAD_BF16 && Y->dtype == RAD_BF16)      mode = 0;
    else if (X->dtype == RAD_BF16 && Y->dtype == RAD_F32)  mode = 1;
    else if (X->dtype == RAD_F32 && Y->dtype == RAD_BF16)  mode = 2;
    else return RAD_E_DTYPE;

    const int64_t nel = rad_tensor_numel(Y);
    if (nel <= 0 || rad_tensor_numel(X) < nel) return RAD_E_SHAPE;
    /* Rank > 2 is refused for the same reason as in act_mul: ref's offlin decomposes the
     * linear index over every dimension's own stride; one pitch per tensor cannot express it. */
    if (X->rank > 2 || Y->rank > 2) return RAD_E_UNSUPPORTED;
    if (Y->rank >= 2 && Y->stride[Y->rank - 1] != 1) return RAD_E_STRIDE;
    if (X->rank >= 2 && X->stride[X->rank - 1] != 1) return RAD_E_STRIDE;

    /* The column extent for the pitch split: the declared `n`, else y's last dimension. */
    long long n = 0;
    if (!rad_args_geti(a, "n", &n) || n <= 0) n = rad_tensor_lastdim(Y);
    if (n <= 0) n = nel;                       /* rank-1 pair: one row of nel */
    if (nel % n != 0) return RAD_E_SHAPE;

    const int64_t x_pitch = X->rank >= 2 ? X->stride[X->rank - 2] : n;
    const int64_t y_pitch = Y->rank >= 2 ? Y->stride[Y->rank - 2] : n;
    const int contig = x_pitch == n && y_pitch == n;

    const uint32_t blocks = (uint32_t)((nel + kThreads - 1) / kThreads);
    if (blocks == 0 || blocks > 0x7fffffffu) return RAD_E_SHAPE;

    hipStream_t stream = (hipStream_t)(uintptr_t)s;
    #define R3D_CAST_LAUNCH(M) do {                                            \
        if (contig)                                                            \
            hipLaunchKernelGGL((cast_kernel<M, 1>), dim3(blocks),              \
                               dim3(kThreads), 0, stream,                      \
                               X->data, Y->data, (uint64_t)nel, (uint32_t)n,   \
                               x_pitch, y_pitch);                              \
        else                                                                   \
            hipLaunchKernelGGL((cast_kernel<M, 0>), dim3(blocks),              \
                               dim3(kThreads), 0, stream,                      \
                               X->data, Y->data, (uint64_t)nel, (uint32_t)n,   \
                               x_pitch, y_pitch);                              \
    } while (0)
    if (mode == 0)      R3D_CAST_LAUNCH(0);
    else if (mode == 1) R3D_CAST_LAUNCH(1);
    else                R3D_CAST_LAUNCH(2);
    #undef R3D_CAST_LAUNCH
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
