/* r3d_attn.h -- internal hand-off between the attn_paged entry TU and the WMMA tile TU.
 *
 * Mirrors r3d_gemm.h: the entry (r3d_attn_paged.hip) derives every scalar exactly where
 * libref derives it, fills R3dAttnParams, and offers it to the WMMA launcher
 * (r3d_attn_paged_wmma.hip). RAD_OK from the launcher means the tile was enqueued; any
 * other code means "not me" and the entry launches its own correctness-first kernel,
 * which was never launched in that case. Plain C++ linkage within the plugin .so; not
 * part of the plugin's export contract.
 */
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

/* One launch's scalars. The v1 kernel and the WMMA tile read the SAME struct the entry
 * computed from the tensor headers, so both paths see identical pitches, caps, and scale
 * (the dispatch decision itself can never fork the arithmetic it is supposed to A/B). */
struct R3dAttnParams {
    const void* q;      const void* kv;      void* out;   /* out typed by out_f32 */
    const void* bt;     const void* su;      const void* cus;
    const RadTensor* kdes;  const RadTensor* vdes;   /* nullable */
    uint32_t bt_dt, su_dt, cus_dt;
    int out_f32;
    uint32_t qdt, kvdt;                     /* operand dtypes: the WMMA gate needs them */

    int64_t q_rs, q_hs, q_ds;
    int64_t o_rs, o_hs, o_ds;
    int64_t kv_sb, kv_sh, kv_so, kv_sd;     /* kv_view: block, kv head, slot, element */
    int64_t bt_rs, bt_cs;

    int64_t head_dim, gqa, kv_heads, q_heads, n_seq;
    int64_t block_size, jcap;               /* jcap = max_blocks * block_size */
    int64_t max_blocks;
    int64_t rows;                           /* query rows: numel(q)/(q_heads*head_dim) */
    int     causal;
    int64_t window;
    float   scale;
    int64_t q_len_uniform;                  /* rows/n_seq when cu_seqlens is absent */

    /* split-KV decode only (r3d_attn_paged_decode.hip); zero/null for every other path.
     * pf is the engine's scratch arena (RadArgs.scratch), laid out as
     *   float  O   [(tok*q_heads + h) * splits + sp][head_dim]   the un-normalized O
     *   float  m   [same idx]                                    the slice's row max
     *   float  l   [same idx]                                    its exp-sum
     * (the m/l planes sit after the O plane, like libr4d's non-PF16 partials). */
    void*   pf;
    int     splits;
};

/* Launches the WMMA flash tile. Returns RAD_OK when enqueued, or RAD_E_UNSUPPORTED /
 * RAD_E_DEVICE to hand the launch back to the entry's v1 kernel. */
int r3d_attn_paged_wmma_launch(const R3dAttnParams& P, hipStream_t stream);

/* Head_dim-256 tile sizing (issue #11 milestone 2). The 256 instantiation cannot use
 * static __shared__ (the fitted carve exceeds the 48 KiB static ceiling), so the kernel
 * reads its LDS through extern __shared__ and the LAUNCHER must pass the exact byte
 * count -- but Carve lives inside each TU's chip gate while the launcher lives outside
 * it. This host-side constexpr is the launcher's copy of the Carve<256,PAD>::total
 * formula; every 256 instantiation carries a static_assert pinning it to the kernel's
 * own C::total, so the two can never drift (mirrors the kvvec-gate-duplicated-
 * verbatim pattern of the DBUF dispatch). PAD selects the bank-pad A/B arms:
 * 0 = S pitch kBN+4, P pitch kBN+2; 1 = S pitch kBN+1 (bank-spread variant). Pads are
 * addressing-only: every PAD produces bit-identical output; the bench picks the speed.
 * D must be 256; other head_dims keep the historical static path and never call this. */
constexpr int r3d_attn256_smem(int pad) {
    constexpr int bm = 64, bn = 32;                       /* kBM, kBN (body.inc law) */
    const int kQP = 256 + 8, kVP = 256 + 8, kPP = bn + 2; /* Q/KV row pitch 8-elem pad:
                                                            * uint4 LDS stores need 16 B row
                                                            * starts (256+4=520 B breaks it) */
    const int kSP = pad ? bn + 1 : bn + 4;
    auto al16 = [](int x) { return (x + 15) & ~15; };
    const int offQ  = 0;
    const int offS  = al16(offQ + bm * kQP * 2);
    const int offP  = al16(offS + bm * kSP * 4);
    const int offKV = al16(offP + bm * kPP * 2);
    const int offM  = al16(offKV + bn * kVP * 2);
    const int offRB = offM + 2 * bm * 4;
    const int offCB = offRB + bn * 8;
    const int offRB2 = offCB + bn;         /* DBUF addressing pair (8-aligned) */
    const int offCB2 = offRB2 + bn * 8;
    return offCB2 + bn + 16;
}

/* The scalar reference port (r3d_attn_paged.hip's v1 kernel), exposed so the decode row can
 * serve what the split-K tile refuses (foreign chips, R3D_NO_WMMA, head_dim outside the
 * WMMA templates) without re-deriving a single scalar. */
int r3d_attn_v1_launch(const R3dAttnParams& P, hipStream_t stream);

/* Derives every scalar of R3dAttnParams from the RadArgs exactly where libref derives it
 * (lives in r3d_attn_paged.hip). All three attn_paged entries call this one function, so the
 * dispatch decision can never fork the arithmetic it is supposed to A/B. Returns RAD_OK or
 * the entry's refusal code; fills P on RAD_OK only. */
int r3d_attn_fill(const RadArgs* a, R3dAttnParams& P);
