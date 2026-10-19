/* r3d_actmul.hip -- libr3d's `act_mul` on RDNA3: the gated SwiGLU/GLU activation.

 * The arithmetic is libref's (ref_elementwise.cpp ref_act_mul + ref_common.h act_*, Apache-2.0,
 * (c) radiance authors -- ported with attribution), with the rounding in the same places:
 *   * the activation value is rounded to y's dtype BEFORE the product (two roundings total:
 *     act(a) -> bf16-as-f32, then the product -> bf16), exactly as the schema states;
 *   * the transcendentals widen to double like the oracle's (exp/erf/tanh in double, rounded to
 *     f32 once), so every f32 arithmetic step is IEEE-exact and only the two libm-vs-libdevice
 *     double results carry a possible sub-ULP difference -- invisible under bf16 rounding;
 *   * `-ffp-contract=off` (CMakeLists) keeps each multiply a separate RTNE step.
 *
 * The double-width transcendentals cost ALU time on a consumer part (RDNA3 f64 is narrow), but
 * the op is bandwidth-bound on real shapes -- three bf16 tensors of rows*n streamed per token --
 * so the first port takes bit-faithfulness over speed. A fast sincosf/expf-shaped row can be
 * added later if kbench says the tolerance absorbs it; that is an optimization decision, not a
 * default.
 *
 * Strides: a, b and y each live at their own row pitch (the schema's column-slice note), so
 * three pitches are read off the operands; a non-unit element stride is refused at launch
 * rather than walked, matching r3d_embed.hip's stance.
 */

#include <cstring>

#include <hip/hip_runtime.h>

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
/* The oracle's round_dt(dtype, v) for bf16: the f32 value rounded to bf16 precision but kept
 * as f32 -- the schema's "rounded ... before the product". */
__device__ __forceinline__ float r3d_round_bf16(float f) {
    return r3d_bf16_to_f32(r3d_f32_to_bf16(f));
}

/* The four oracle activations, each ref_common.h's expression with the double-width
 * transcendental. act index order is ref's kActs: silu, gelu, gelu_tanh, sigmoid. */
__device__ __forceinline__ float r3d_act(int act, float x) {
    switch (act) {
        case 0: {                                     /* silu */
            const float e = (float)exp(-(double)x);
            return x * (1.0f / (1.0f + e));
        }
        case 1:                                       /* gelu, the exact erf form */
            return 0.5f * x * (1.0f + (float)erf((double)(x * 0.70710678118654752f)));
        case 2: {                                     /* gelu_tanh */
            const float t = (float)tanh((double)(0.7978845608028654f * (x + 0.044715f * x * x * x)));
            return 0.5f * x * (1.0f + t);
        }
        default: {                                    /* sigmoid */
            const float e = (float)exp(-(double)x);
            return 1.0f / (1.0f + e);
        }
    }
}

__global__ __launch_bounds__(kThreads) void act_mul_kernel(
    const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, uint16_t* __restrict__ y,
    uint64_t total, uint32_t n, int act,
    int64_t a_pitch, int64_t b_pitch, int64_t y_pitch) {
    const uint64_t idx = (uint64_t)blockIdx.x * kThreads + threadIdx.x;
    for (uint64_t e = idx; e < total; e += (uint64_t)gridDim.x * kThreads) {
        const uint64_t r = e / n, i = e - r * n;
        const float g = r3d_bf16_to_f32(a[r * a_pitch + i]);
        const float v = r3d_act(act, g);
        const uint16_t vb = r3d_f32_to_bf16(v);          /* the schema's first rounding */
        const uint16_t bv = b[r * b_pitch + i];
        y[r * y_pitch + i] = r3d_f32_to_bf16(r3d_bf16_to_f32(vb) * r3d_bf16_to_f32(bv));
    }
}

} /* anonymous namespace */

/* Launch entry, operands positional in the act_mul schema: a (gate), b, y. */
extern "C" int r3d_rad_act_mul(const RadArgs* a, RadStream s) {
    const RadTensor* A = rad_arg_in(a, 0);
    const RadTensor* B = rad_arg_in(a, 1);
    const RadTensor* Y = rad_arg_in(a, 2);
    if (!A || !B || !Y) return RAD_E_INVAL;
    if (A->dtype != RAD_BF16 || B->dtype != RAD_BF16 || Y->dtype != RAD_BF16)
        return RAD_E_DTYPE;

    /* ref's kActs order; the row constrains act to "silu" but cold callers can ask for any. */
    static const char* const kActs[] = { "silu", "gelu", "gelu_tanh", "sigmoid" };
    long long ai = -1;
    const char* as = rad_args_gets(a, "act");
    if (as) for (int k = 0; k < 4; ++k) if (!std::strcmp(as, kActs[k])) ai = k;
    if (ai < 0) return RAD_E_INVAL;

    /* n: the declared parameter, else numel(Y) (ref's default). */
    long long n = 0;
    if (!rad_args_geti(a, "n", &n) || n <= 0) n = rad_tensor_numel(Y);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rad_tensor_numel(Y) / n;
    if (rows <= 0 || rad_tensor_numel(A) < rows * n || rad_tensor_numel(B) < rows * n)
        return RAD_E_SHAPE;

    /* Unit element stride required; each tensor keeps its own row pitch. */
    if (Y->rank >= 2 && Y->stride[Y->rank - 1] != 1) return RAD_E_STRIDE;
    if ((A->rank >= 2 && A->stride[A->rank - 1] != 1) ||
        (B->rank >= 2 && B->stride[B->rank - 1] != 1)) return RAD_E_STRIDE;
    if (rows > 1 && (A->rank < 2 || B->rank < 2 || Y->rank < 2)) return RAD_E_SHAPE;
    const int64_t a_pitch = A->rank >= 2 ? A->stride[A->rank - 2] : n;
    const int64_t b_pitch = B->rank >= 2 ? B->stride[B->rank - 2] : n;
    const int64_t y_pitch = Y->rank >= 2 ? Y->stride[Y->rank - 2] : n;

    const uint64_t total = (uint64_t)rows * (uint64_t)n;
    const uint32_t blocks = (uint32_t)((total + kThreads - 1) / kThreads);
    if (blocks == 0 || blocks > 0x7fffffffu) return RAD_E_SHAPE;

    hipStream_t stream = (hipStream_t)(uintptr_t)s;
    hipLaunchKernelGGL(act_mul_kernel, dim3(blocks), dim3(kThreads), 0, stream,
                       (const uint16_t*)A->data, (const uint16_t*)B->data, (uint16_t*)Y->data,
                       total, (uint32_t)n, (int)ai, a_pitch, b_pitch, y_pitch);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
