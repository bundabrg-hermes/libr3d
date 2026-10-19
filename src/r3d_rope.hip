/* r3d_rope.hip -- libr3d's `rope` on RDNA3: rotary embedding in place over a fused qkv row.

 * The arithmetic is libref's (ref_attn.cpp ref_rope / rope_freqs / rope_head / rope_pos,
 * Apache-2.0, (c) radiance authors -- ported with attribution), with the rounding in the same
 * places:
 *   * the frequency ladder is the oracle's width-for-width: 1.0f / (float)pow((double)theta,
 *     (double)((float)(2i) / (float)rot)) -- the exponent is an f32 divide widened to double,
 *     the result narrowed to f32 and inverted there, exactly like ref_pow's double path;
 *   * the angle is pos * inv in f32 and the pair rotates with the accurate sincosf in f32 --
 *     the same choice libr4d's row documents ("fp32 angles through the accurate sincosf"):
 *     sincosf vs (float)cos(double) differs by ~1 f32-ULP, invisible under the bf16 store;
 *   * the two products and the difference are separate f32 steps (-ffp-contract=off) and the
 *     store is the single RTNE narrowing.
 *
 * Scope: the neox and gptj pairings over [M] positions, which is everything a bf16 text model
 * issues (neox). The multi-component modes (mrope/imrope/axial), a non-rotary ladder, yarn and
 * llama3 frequency scaling, and [C, M] position tensors are refused at launch, not silently
 * approximated -- and the row's constraints keep the selector from asking.
 *
 * Layout: one block per token; the frequency ladder is built once per block in LDS (the heads
 * within a token share it), then threads stride the (head, pair) work of the Q and K heads.
 * The V heads that follow are untouched.
 */

#include <cstring>

#include <hip/hip_runtime.h>

#include "rad_abi.h"
#include "rad_plugin.h"

namespace {

constexpr uint32_t kThreads = 256;
constexpr int kGptj = 1;   /* ref's kModes index for gptj; 0 is neox */

__device__ __forceinline__ float r3d_bf16_to_f32(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}
__device__ __forceinline__ uint16_t r3d_f32_to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* One token row: build the ladder in shared, then rotate `heads` = n_head + n_head_kv heads,
 * one (head, pair) item per striding thread. `Tok` is the positions dtype (the schema's i32;
 * i64 and f32 -- a fractional coordinate's own dtype -- are accepted for cold callers). */
template <typename Tok>
__global__ __launch_bounds__(kThreads) void rope_kernel(
    uint16_t* __restrict__ qkv, const Tok* __restrict__ pos,
    int64_t row_pitch, int64_t ds, uint32_t half, uint32_t rot, uint32_t heads, int mode,
    float theta, float div) {
    extern __shared__ float inv[];
    const int64_t m = blockIdx.x;

    /* The ladder, ref_rope_freqs' expression at the oracle's widths. The row refuses any
     * ladder but `rotary`, whose denominator is rot itself. */
    for (uint32_t i = threadIdx.x; i < half; i += kThreads)
        inv[i] = 1.0f / (float)pow((double)theta, (double)((float)(2u * i) / (float)rot));
    __syncthreads();

    /* rope_pos: the position's value divided by the position scale in f32. An integer
     * operand contributes its (float) value; a float operand its own. */
    const float p = (float)(double)pos[m] / div;

    uint16_t* __restrict__ xr = qkv + m * row_pitch;
    const uint32_t work = heads * half;
    for (uint32_t w = threadIdx.x; w < work; w += kThreads) {
        const uint32_t h = w / half, i = w - h * half;
        const uint32_t i0 = (mode == kGptj) ? 2u * i : i;
        const uint32_t i1 = (mode == kGptj) ? 2u * i + 1u : i + half;
        const float ang = p * inv[i];
        float c, s;
        sincosf(ang, &s, &c);                    /* magnitude is 1 outside yarn, which is refused */
        const float x0 = r3d_bf16_to_f32(xr[i0 * ds]);
        const float x1 = r3d_bf16_to_f32(xr[i1 * ds]);
        xr[i0 * ds] = r3d_f32_to_bf16(x0 * c - x1 * s);
        xr[i1 * ds] = r3d_f32_to_bf16(x0 * s + x1 * c);
    }
}

} /* anonymous namespace */

/* Launch entry. Operands positional in the rope schema: qkv (INOUT), positions (IN). */
extern "C" int r3d_rad_rope(const RadArgs* a, RadStream s) {
    const RadTensor* qkv = rad_arg_in(a, 0);   /* INOUT role: read as descriptor, written via data */
    const RadTensor* pos = rad_arg_in(a, 1);
    if (!qkv || !pos) return RAD_E_INVAL;
    if (qkv->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (pos->dtype != RAD_I32 && pos->dtype != RAD_I64 && pos->dtype != RAD_F32)
        return RAD_E_DTYPE;

    long long head_dim = 0, n_head = 0, n_head_kv = 0, rot = 0;
    if (!rad_args_geti(a, "head_dim", &head_dim) || head_dim <= 0) return RAD_E_SHAPE;
    if (!rad_args_geti(a, "n_head", &n_head) || n_head <= 0) return RAD_E_SHAPE;
    if (!rad_args_geti(a, "n_head_kv", &n_head_kv) || n_head_kv < 0) return RAD_E_SHAPE;
    rot = rad_args_geti_or(a, "rotary_dim", head_dim);
    if (rot <= 0 || rot > head_dim || (rot & 1)) return RAD_E_SHAPE;

    /* Modes, ref's kModes order; the row serves neox (0) and gptj (1) only. */
    const char* ms = rad_args_gets(a, "mode");
    if (!ms) return RAD_E_INVAL;
    const int mode = !std::strcmp(ms, "neox") ? 0 : !std::strcmp(ms, "gptj") ? 1 : -1;
    if (mode < 0) return RAD_E_UNSUPPORTED;

    /* Refusals, each a shape the oracle computes differently than this kernel: the oracle
     * ignores sections and yarn for single-component neox/gptj too, but a mistyped ladder or
     * a scaling request reaching this row would compute silently different angles. */
    if (pos->rank >= 2) return RAD_E_UNSUPPORTED;          /* multi-component positions */
    if (rad_args_gets(a, "sections") || rad_args_gets(a, "ladder")) return RAD_E_UNSUPPORTED;
    if (rad_args_getf_or(a, "yarn_factor", 0.0) != 0.0 ||
        rad_args_getf_or(a, "llama3_factor", 0.0) != 0.0) return RAD_E_UNSUPPORTED;

    const float theta = (float)rad_args_getf_or(a, "theta", 10000.0);
    const float scale = (float)rad_args_getf_or(a, "scale", 1.0);
    const float div = scale != 0.0f ? scale : 1.0f;

    const int64_t M = rad_tensor_numel(pos);
    const int64_t stride_row = (n_head + 2 * n_head_kv) * head_dim;
    if (M <= 0 || M > 0x7fffffffLL) return RAD_E_SHAPE;
    if (rad_tensor_numel(qkv) < M * stride_row) return RAD_E_SHAPE;
    if (qkv->rank >= 2 && qkv->stride[qkv->rank - 1] != 1) return RAD_E_STRIDE;

    /* The token pitch, mirroring ref_gemm.h's row_stride(qkv, stride_row) exactly: fold the
     * trailing dimensions until they span the row, and the stride one axis out carries it.
     * For a rank-3 [M, heads, head_dim] qkv -- what the q-only (n_head_kv=0) geometries
     * present -- that is stride[0]; a bare stride[rank-2] there would read every token at
     * the head stride. */
    int64_t row_pitch;
    if (qkv->rank < 2) {
        row_pitch = stride_row;
    } else {
        int64_t span = 1;
        int hit = -1;
        for (int i = (int)qkv->rank - 1; i >= 0; --i) {
            span *= qkv->shape[i];
            if (span == stride_row) { hit = i; break; }
            if (span > stride_row) break;
        }
        row_pitch = hit >= 0 ? (hit > 0 ? qkv->stride[hit - 1] : stride_row)
                             : qkv->stride[qkv->rank - 2];   /* shapes disagree: ref's fallback */
    }
    const int64_t ds = 1;   /* element stride, enforced unit above; ref reads laststride(qkv) */

    const uint32_t half = (uint32_t)(rot / 2);
    const uint32_t heads = (uint32_t)(n_head + n_head_kv);
    const size_t shmem = (size_t)half * sizeof(float);

    hipStream_t stream = (hipStream_t)(uintptr_t)s;
    if (pos->dtype == RAD_I32)
        hipLaunchKernelGGL((rope_kernel<int32_t>), dim3((uint32_t)M), dim3(kThreads),
                           shmem, stream, (uint16_t*)qkv->data, (const int32_t*)pos->data,
                           row_pitch, ds, half, (uint32_t)rot, heads, mode, theta, div);
    else if (pos->dtype == RAD_I64)
        hipLaunchKernelGGL((rope_kernel<int64_t>), dim3((uint32_t)M), dim3(kThreads),
                           shmem, stream, (uint16_t*)qkv->data, (const int64_t*)pos->data,
                           row_pitch, ds, half, (uint32_t)rot, heads, mode, theta, div);
    else
        hipLaunchKernelGGL((rope_kernel<float>), dim3((uint32_t)M), dim3(kThreads),
                           shmem, stream, (uint16_t*)qkv->data, (const float*)pos->data,
                           row_pitch, ds, half, (uint32_t)rot, heads, mode, theta, div);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
