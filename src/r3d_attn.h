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
};

/* Launches the WMMA flash tile. Returns RAD_OK when enqueued, or RAD_E_UNSUPPORTED /
 * RAD_E_DEVICE to hand the launch back to the entry's v1 kernel. */
int r3d_attn_paged_wmma_launch(const R3dAttnParams& P, hipStream_t stream);
