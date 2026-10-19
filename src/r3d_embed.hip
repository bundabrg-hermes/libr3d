/* r3d_embed.hip -- libr3d's `embed_lookup` on RDNA3: the embedding gather.

 * The arithmetic is libref's (ref_gemm.cpp embed_gather, Apache-2.0, (c) radiance authors --
 * ported with attribution), with the rounding in the same places:
 *   * the gathered bf16 value widens to f32, `mult` (Gemma's sqrt(hidden)) scales it there,
 *     and the store is the single RTNE narrowing (identical conversions to r3d_rmsnorm.hip);
 *   * a token id outside [vocab_offset, vocab_offset + n_vocab) -- which includes every
 *     negative id -- writes a zero row, so a vocabulary-sharded table sums correctly under a
 *     later all_reduce;
 *   * `-ffp-contract=off` (CMakeLists) keeps the multiply an exact identity when mult == 1.
 *
 * One workgroup per token, threads striding the row: a gather is bandwidth-bound and the
 * vocabulary row is contiguous, so this is a copy with an indirection, not arithmetic. The
 * row table's `dtype` constraint keeps the core from selecting this row for anything but a
 * bf16 table; the entry point re-checks for cold callers (kbench, selftest).
 *
 * Unlike ref, this row refuses a table that is not contiguous within its row rather than
 * walking arbitrary element strides: the engine's tables are row-major, and the check happens
 * at resolve so a refusal is an error, never wrong numbers.
 */

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

/* One block per token row of x. `Tok` is the token tensor's integer type (the schema's i32;
 * i64 is accepted for cold callers that build tensors by hand). */
template <typename Tok>
__global__ __launch_bounds__(kThreads) void embed_lookup_kernel(
    const Tok* __restrict__ tok, const uint16_t* __restrict__ wte, uint16_t* __restrict__ x,
    uint32_t n_embd, int64_t n_vocab, int64_t vocab_offset, float mult,
    int64_t wte_pitch, int64_t x_pitch) {
    const int64_t m = blockIdx.x;
    const int64_t id = (int64_t)tok[m] - vocab_offset;
    uint16_t* __restrict__ xr = x + m * x_pitch;

    if (id < 0 || id >= n_vocab) {          /* padding row, or another rank's token */
        for (uint32_t i = threadIdx.x; i < n_embd; i += kThreads) xr[i] = 0;
        return;
    }
    const uint16_t* __restrict__ wr = wte + id * wte_pitch;
    for (uint32_t i = threadIdx.x; i < n_embd; i += kThreads)
        xr[i] = r3d_f32_to_bf16(r3d_bf16_to_f32(wr[i]) * mult);
}

} /* anonymous namespace */

/* Launch entry, operands positional in libref's embed_lookup schema order: tokens, wte, x. */
extern "C" int r3d_rad_embed_lookup(const RadArgs* a, RadStream s) {
    const RadTensor* tok = rad_arg_in(a, 0);
    const RadTensor* wte = rad_arg_in(a, 1);
    const RadTensor* x   = rad_arg_in(a, 2);
    if (!tok || !wte || !x) return RAD_E_INVAL;
    if (x->dtype != RAD_BF16 || wte->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (tok->dtype != RAD_I32 && tok->dtype != RAD_I64) return RAD_E_DTYPE;

    /* n_embd / n_vocab: the declared parameters, else the table's extents (ref's rule). */
    long long n_embd = 0, n_vocab = 0;
    if (!rad_args_geti(a, "n_embd", &n_embd) || n_embd <= 0)
        n_embd = rad_tensor_lastdim(wte);
    if (!rad_args_geti(a, "n_vocab", &n_vocab) || n_vocab <= 0)
        n_vocab = wte->rank >= 2 ? wte->shape[wte->rank - 2] : 0;
    if (n_embd <= 0 || n_vocab <= 0) return RAD_E_SHAPE;
    long long voff = 0;
    if (rad_args_geti(a, "vocab_offset", &voff) && voff < 0) return RAD_E_SHAPE;

    const double mult = rad_args_getf_or(a, "mult", 1.0);
    /* ref: M = numel(x) / n_embd. rad_tensor_rows counts every leading axis as a row, so it
     * is numel/n_embd for rank >= 2 and needs no special case. */
    const int64_t M = rad_tensor_rows(x);
    if (M <= 0 || rad_tensor_numel(tok) < M) return RAD_E_SHAPE;

    /* The row's element extent is n_embd and the table's rows are contiguous; its row pitch
     * may not be (a table sliced from a wider allocation is fine). */
    if (rad_tensor_lastdim(x) != n_embd) return RAD_E_SHAPE;
    if (wte->rank < 2 || wte->stride[wte->rank - 1] != 1) return RAD_E_STRIDE;
    if (x->rank > 1 && x->stride[x->rank - 1] != 1) return RAD_E_STRIDE;
    if (rad_tensor_numel(wte) < n_vocab * n_embd) return RAD_E_SHAPE;
    const int64_t wte_pitch = wte->stride[wte->rank - 2];
    const int64_t x_pitch   = x->rank >= 2 ? rad_tensor_row_pitch(x) : n_embd;
    if (M > 0x7ffffffLL) return RAD_E_SHAPE;

    hipStream_t stream = (hipStream_t)(uintptr_t)s;
    if (tok->dtype == RAD_I32)
        hipLaunchKernelGGL((embed_lookup_kernel<int32_t>), dim3((uint32_t)M), dim3(kThreads),
                           0, stream, (const int32_t*)tok->data, (const uint16_t*)wte->data,
                           (uint16_t*)x->data, (uint32_t)n_embd, n_vocab, voff, (float)mult,
                           wte_pitch, x_pitch);
    else
        hipLaunchKernelGGL((embed_lookup_kernel<int64_t>), dim3((uint32_t)M), dim3(kThreads),
                           0, stream, (const int64_t*)tok->data, (const uint16_t*)wte->data,
                           (uint16_t*)x->data, (uint32_t)n_embd, n_vocab, voff, (float)mult,
                           wte_pitch, x_pitch);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
