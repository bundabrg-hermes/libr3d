/* r3d_embed.hip -- libr3d's embedding gathers on RDNA3: `embed_lookup` (the bf16 table) and
 * `embed_lookup_q` (the quantised table: the block-fp8 embedding's device arm, Q61b).

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
static int r3d_embed_lookup_impl(const RadArgs*, RadStream);
extern "C" int r3d_rad_embed_lookup(const RadArgs* a, RadStream s)
    R3D_DUMP_WRAP("embed_lookup", r3d_embed_lookup_impl)
static int r3d_embed_lookup_impl(const RadArgs* a, RadStream s) {
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

/* ============================================================================================
 * embed_lookup_q (Q61b) -- the gather from a QUANTISED table: the block-fp8 embedding's
 * RDNA3 device arm (qwen35_fp8's token_embd: E4M3 codes with a bf16 scale per [128 x 128]
 * tile, declared as the codes + scale views of one stored weight) and the n-gram table's
 * one-scale form. The walk and the entry are libr4d's r4d_model_bf16.hip
 * embed_lookup_q_kernel ported to RDNA3 -- v_cvt_pk_f32_fp8 does not exist on this family
 * (the Q62 ledger law), so the software decode below stands in for it.
 *
 * The arithmetic is libref's (ref_gemm.cpp embed_gather / ref_embed_lookup_q, Apache-2.0,
 * (c) radiance authors -- ported with attribution), bit-equal in the Q62 sense:
 *   * an E4M3 code decodes to f32 EXACTLY -- the loop-free decode below is the fp8a8/logits/
 *     mxfp4 TUs' proven bit-equivalent of the abi's rad_fp8e4m3_to_f32, which is libref
 *     ld_dt's decode;
 *   * the scale widens to f32 exactly: the per-tile entry of the block plane
 *     (bf16 [ceil(n_vocab/blk_r)][ceil(n_embd/blk_c)], one entry per [blk_r x blk_c] tile,
 *     or its f32 spelling), or the whole-table scalar, an absent scale being 1;
 *   * the product rounds ONCE in f32 -- and indeed never rounds: 4 significant code bits
 *     (e4m3) times 8 (bf16 scale) is at most 12 significand bits, exact in f32 -- so only
 *     the store narrows, the single RTNE below, whose NaN form matches rad_f32_to_bf16
 *     that libref stores through (see r3d_rmsnorm.hip for the conversion's provenance);
 *   * an id outside [0, n_vocab) after `vocab_offset` -- which includes every negative id --
 *     writes a zero row: ref's have_row test, embed_lookup's rule above, and the device
 *     rule of docs/OPS.md (the kernel cannot read its ids before it runs).
 *
 * One thread per quad of entries when the width, pitches and bases allow the quad's 4- and
 * 8-byte accesses (the n-gram row is 160 entries -- a block per row would idle the machine;
 * the same walk serves the 5120-wide table), one entry per thread otherwise. The block
 * plane's tile index uses this rank's LOCAL id: a vocab shard owns the scale rows of its
 * codes rows, exactly as the shard owns its codes rows.
 */

namespace {

/* rad_plugin.h rad_fp8e4m3_to_f32, loop-free form -- the fp8a8 TU's gemv decoder ported
 * (bit-equivalence to the oracle's while-loop decode is that TU's proven claim, and the
 * logits and mxfp4 TUs carry the same copy for the same reason: the oracle decode is
 * host-only linkage, and a divergent while-loop has no place in a bandwidth row's load
 * path). */
__device__ __forceinline__ float fp8e4m3_gemv(uint8_t b) {
    const uint32_t s = (uint32_t)(b >> 7) << 31;
    const uint32_t e = (uint32_t)(b >> 3) & 0xfu, m = (uint32_t)b & 0x7u;
    uint32_t u;
    if (e == 0u) {
        if (m == 0u) return __uint_as_float(s);
        const float f = (float)m / 512.0f;                                /* m * 2^-9 */
        u = s | (__float_as_uint(f) & 0x7fffffffu);
    } else if (e == 0xfu && m == 0x7u) {
        u = s | 0x7fc00000u;                                              /* the NaN */
    } else {
        u = s | ((e - 7u + 127u) << 23) | (m << 20);
    }
    return __uint_as_float(u);
}

/* `Q` names the E4M3 codes table (else a plain bf16 table -- the n-gram gather's in-place
 * plain form, which takes this op too); `B` the per-tile plane (else the whole-table
 * scalar). V = 4 walks a quad, V = 1 one entry, per the entry point's alignment gate.
 * `Tok` is the token tensor's integer type (the schema's i32; i64 is accepted for cold
 * callers that build tensors by hand, as the bf16 row above does). */
template <typename Tok, uint32_t V, bool Q, bool B>
__global__ __launch_bounds__(kThreads) void embed_lookup_q_kernel(
    const Tok* __restrict__ tok, const void* __restrict__ wte, const void* __restrict__ sc,
    uint32_t sc_f32, uint32_t sc_cols, uint32_t blk_r, uint32_t blk_c,
    uint16_t* __restrict__ x, uint32_t M, uint32_t n_embd, int64_t n_vocab,
    int64_t vocab_off, int64_t wte_pitch, int64_t x_pitch) {
    const uint32_t per_row = n_embd / V;
    const uint64_t g = (uint64_t)blockIdx.x * kThreads + threadIdx.x;
    if (g >= (uint64_t)M * per_row) return;
    const uint32_t m = (uint32_t)(g / per_row), j = (uint32_t)(g % per_row) * V;
    const int64_t id = (int64_t)tok[m] - vocab_off;
    uint16_t* __restrict__ dst = x + (size_t)m * x_pitch + j;
    if (id < 0 || id >= n_vocab) {              /* padding row, or another rank's token */
        if constexpr (V == 4) *(uint2*)dst = make_uint2(0u, 0u);
        else                  dst[0] = 0;
        return;
    }
    float s;                                    /* one scale for the table, or one for this tile */
    if constexpr (B)
        s = sc_f32 ? ((const float*)sc)[(size_t)(id / (int64_t)blk_r) * sc_cols + j / blk_c]
                   : r3d_bf16_to_f32(
                         ((const uint16_t*)sc)[(size_t)(id / (int64_t)blk_r) * sc_cols + j / blk_c]);
    else
        s = !sc ? 1.0f : sc_f32 ? *(const float*)sc : r3d_bf16_to_f32(*(const uint16_t*)sc);
    const size_t at = (size_t)id * (size_t)wte_pitch + j;
    float v[V];
    if constexpr (Q && V == 4) {
        const uint32_t q = *(const uint32_t*)((const uint8_t*)wte + at);
        v[0] = fp8e4m3_gemv((uint8_t)q);       v[1] = fp8e4m3_gemv((uint8_t)(q >> 8));
        v[2] = fp8e4m3_gemv((uint8_t)(q >> 16)); v[3] = fp8e4m3_gemv((uint8_t)(q >> 24));
    } else if constexpr (Q) {
        v[0] = fp8e4m3_gemv(((const uint8_t*)wte)[at]);
    } else if constexpr (V == 4) {
        const uint2 b = *(const uint2*)((const uint16_t*)wte + at);
        v[0] = r3d_bf16_to_f32((uint16_t)b.x); v[1] = r3d_bf16_to_f32((uint16_t)(b.x >> 16));
        v[2] = r3d_bf16_to_f32((uint16_t)b.y); v[3] = r3d_bf16_to_f32((uint16_t)(b.y >> 16));
    } else {
        v[0] = r3d_bf16_to_f32(((const uint16_t*)wte)[at]);
    }
    if constexpr (V == 4) {
        *(uint2*)dst = make_uint2(
            (uint32_t)r3d_f32_to_bf16(v[0] * s) | ((uint32_t)r3d_f32_to_bf16(v[1] * s) << 16),
            (uint32_t)r3d_f32_to_bf16(v[2] * s) | ((uint32_t)r3d_f32_to_bf16(v[3] * s) << 16));
    } else {
        dst[0] = r3d_f32_to_bf16(v[0] * s);
    }
}

} /* anonymous namespace */

/* Launch entry, operands positional in libref's embed_lookup_q schema order (ref_registry's
 * pEmbed/oEmbedQ): tokens, wte, scale (optional; absent is 1), x, ahead (optional; the next
 * prompt's ids, ignored by a kernel that reads its table in place -- OPS.md, as libr4d). */
static int r3d_embed_lookup_q_impl(const RadArgs*, RadStream);
extern "C" int r3d_rad_embed_lookup_q(const RadArgs* a, RadStream s)
    R3D_DUMP_WRAP("embed_lookup_q", r3d_embed_lookup_q_impl)
static int r3d_embed_lookup_q_impl(const RadArgs* a, RadStream s) {
    const RadTensor* tok = rad_arg_in(a, 0);
    const RadTensor* wte = rad_arg_in(a, 1);
    const RadTensor* sc  = rad_arg_in(a, 2);
    const RadTensor* x   = rad_arg_in(a, 3);
    if (!tok || !wte || !x) return RAD_E_INVAL;
    if (x->dtype != RAD_BF16) return RAD_E_DTYPE;
    const bool q = wte->dtype == RAD_F8E4M3;
    if (!q && wte->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (sc && sc->dtype != RAD_BF16 && sc->dtype != RAD_F32) return RAD_E_DTYPE;
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

    const int64_t M = rad_tensor_rows(x);
    if (M <= 0 || rad_tensor_numel(tok) < M) return RAD_E_SHAPE;

    /* The row's element extent is n_embd and the table's rows are contiguous within a row
     * (the same refusal as the bf16 row above: the engine's tables are row-major, and the
     * refusal happens at resolve, never as wrong numbers). */
    if (rad_tensor_lastdim(x) != n_embd) return RAD_E_SHAPE;
    if (wte->rank < 2 || wte->stride[wte->rank - 1] != 1) return RAD_E_STRIDE;
    if (x->rank > 1 && x->stride[x->rank - 1] != 1) return RAD_E_STRIDE;
    if (rad_tensor_numel(wte) < n_vocab * n_embd) return RAD_E_SHAPE;
    const int64_t wte_pitch = wte->stride[wte->rank - 2];
    const int64_t x_pitch   = x->rank >= 2 ? rad_tensor_row_pitch(x) : n_embd;
    if (M > 0x7ffffffLL) return RAD_E_SHAPE;

    /* More than one scale entry is the block form: the plane a "scale" view of a
     * fp8_e4m3*bf16[br x bc] weight hands over, [ceil(n_vocab/br)] rows by
     * [ceil(n_embd/bc)] columns -- ref's block branch derives the tile size the same way
     * from the same two extents, and libr4d's entry is the donor for the checks. One entry
     * is the n-gram's whole-table scale; a bf16 table has no business with a plane. */
    bool block = false;
    uint32_t sc_cols = 0, blk_r = 0, blk_c = 0;
    if (sc && rad_tensor_numel(sc) > 1) {
        block = true;
        if (!q || sc->rank != 2 || sc->stride[sc->rank - 1] != 1 ||
            sc->stride[sc->rank - 2] != sc->shape[sc->rank - 1]) return RAD_E_SHAPE;
        const int64_t sc_rows = sc->shape[sc->rank - 2], sc_c = sc->shape[sc->rank - 1];
        if (sc_rows <= 0 || sc_c <= 0) return RAD_E_SHAPE;
        sc_cols = (uint32_t)sc_c;
        blk_r = (uint32_t)((n_vocab + sc_rows - 1) / sc_rows);
        blk_c = (uint32_t)((n_embd + sc_c - 1) / sc_c);
        if ((int64_t)blk_c > n_embd) return RAD_E_SHAPE;
    }
    const uint32_t sc_f32 = sc && sc->dtype == RAD_F32 ? 1u : 0u;

    /* The quad walk takes four entries a thread; with a block scale it stands down unless
     * the tile cannot split a quad (the block encodings are 128 wide, so it normally
     * holds). The pitches join the gate because the table rows may be sliced: a quad's
     * element address is id*pitch + j with j % 4 == 0. */
    const bool quad = n_embd % 4 == 0 && x_pitch % 4 == 0 && wte_pitch % 4 == 0 &&
                      rad_ptr_aligned(x->data, 8) &&
                      rad_ptr_aligned(wte->data, q ? 4 : 8) &&
                      (!block || blk_c % 4 == 0);
    const uint32_t V = quad ? 4u : 1u;
    const uint64_t threads = (uint64_t)M * (uint64_t)(n_embd / V);
    const uint64_t blocks  = (threads + kThreads - 1) / kThreads;
    if (blocks > 0x7fffffffULL) return RAD_E_SHAPE;
    const dim3 grid((uint32_t)blocks);
    hipStream_t stream = (hipStream_t)(uintptr_t)s;

#define R3D_ELQ(TOKT, VV, QQ, BB)                                                             \
    hipLaunchKernelGGL((embed_lookup_q_kernel<TOKT, VV, QQ, BB>), grid, dim3(kThreads), 0,    \
                       stream, (const TOKT*)tok->data, wte->data,                             \
                       sc ? (const void*)sc->data : nullptr, sc_f32, sc_cols, blk_r, blk_c,   \
                       (uint16_t*)x->data, (uint32_t)M, (uint32_t)n_embd, n_vocab, voff,      \
                       wte_pitch, x_pitch)
#define R3D_ELQ_TOK(TOKT)                                                                     \
    do {                                                                                      \
        if (block) { if (quad) R3D_ELQ(TOKT, 4, true, true); else R3D_ELQ(TOKT, 1, true, true); }  \
        else if (quad) { if (q) R3D_ELQ(TOKT, 4, true, false); else R3D_ELQ(TOKT, 4, false, false); } \
        else       { if (q) R3D_ELQ(TOKT, 1, true, false); else R3D_ELQ(TOKT, 1, false, false); } \
    } while (0)
    if (tok->dtype == RAD_I32) R3D_ELQ_TOK(int32_t);
    else                       R3D_ELQ_TOK(int64_t);
#undef R3D_ELQ_TOK
#undef R3D_ELQ
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
