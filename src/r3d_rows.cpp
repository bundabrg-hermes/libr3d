/* r3d_rows.cpp -- libr3d's declaration: plugin identity and the kernel row table.
 *
 * Pure host C++ including only the ABI, so it compiles without ROCm, exactly like libr4d's
 * r4d_rows.cpp. This skeleton exports the full kernel-plugin contract with an empty table:
 * the loader takes it (ABI matches), finds no device rows for gfx11, and selection behaves as
 * if libr3d were absent. Real rows land here as kernels are implemented; docs/COVERAGE.md
 * tracks the order.
 */

#include "rad_abi.h"

extern "C" {

uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }

const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo info = {
        RAD_PLUGIN_KERNEL,
        "libr3d",
        "0.1.0",
        "radiance kernel library for AMD RDNA3 (gfx11)",
        "gfx1100 gfx1101 gfx1102 gfx1103 gfx11-generic",
    };
    return &info;
}

/* Kernel rows and their op schemas. Every op a plugin implements must have its schema
 * declared by some *loaded kernel plugin* (loader.cpp stages rad_kernel_schema_count/at per
 * plugin; libref is a kbench oracle, not a search-path plugin, so it fixes nothing). On RDNA3
 * libr3d is that declarer, and a kernel whose op has no declared schema is an error:
 *   "implements op 'rmsnorm', which no loaded plugin declares a schema for"
 * The schema below is copied verbatim from libr4d's kSchemas entry, so where libr4d is also
 * present (an RDNA4 box or a mixed rig) the loader's schema_same() check passes and first
 * declarer wins with no refusal. Any future op libr3d implements must likewise arrive with
 * its schema declared here, identical to libr4d's where one exists. */

/* rmsnorm: verbatim libr4d (r4d_rows.cpp pRmsNorm/oRmsNorm). `wadd` is GemmaRMSNorm's `1 +`;
 * `mult` scales the row before its one rounding. `w` optional: absent, the gain is one. */
static const RadParamSpec pRmsNorm[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    { "eps",   RAD_P_F64, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
    { "wadd",  RAD_P_F64, RAD_OPTIONAL },
    { "mult",  RAD_P_F64, RAD_OPTIONAL },
};
static const RadOperandSpec oRmsNorm[] = {
    { "x", RAD_OPD_IN,     0 },
    { "w", RAD_OPD_WEIGHT, 1 },
    { "y", RAD_OPD_OUT,    0 },
};

/* embed_lookup: verbatim libr4d (r4d_rows.cpp pEmbedM/oEmbed). `mult` is Gemma's sqrt(hidden);
 * `vocab_offset` shards the table across TP ranks; `wscale` belongs to embed_lookup_q's
 * quantised table and is unused by this bf16 row. */
static const RadParamSpec pEmbedM[] = {
    { "M",       RAD_P_INT, RAD_REQUIRED },
    { "n_embd",  RAD_P_INT, RAD_REQUIRED },
    { "n_vocab", RAD_P_INT, RAD_REQUIRED },
    { "dtype",   RAD_P_STR, RAD_REQUIRED },
    { "vocab_offset", RAD_P_INT, RAD_OPTIONAL },
    { "wscale",  RAD_P_F64, RAD_OPTIONAL },
    { "mult",    RAD_P_F64, RAD_OPTIONAL },
};
static const RadOperandSpec oEmbed[] = {
    { "tokens", RAD_OPD_IN,     0 },
    { "wte",    RAD_OPD_WEIGHT, 0 },
    { "x",      RAD_OPD_OUT,    0 },
};

/* kv_store: verbatim libr4d (r4d_rows.cpp pKvStore/oKvStore). The scales are the reader's
 * descales and unused for the bf16 cache this row serves. */
static const RadParamSpec pKvStore[] = {
    { "M",          RAD_P_INT, RAD_REQUIRED },
    { "head_dim",   RAD_P_INT, RAD_REQUIRED },
    { "n_head_kv",  RAD_P_INT, RAD_REQUIRED },
    { "block_size", RAD_P_INT, RAD_REQUIRED },
    { "kv_dtype",   RAD_P_STR, RAD_REQUIRED },
    { "k_scale",    RAD_P_F64, RAD_OPTIONAL },
    { "v_scale",    RAD_P_F64, RAD_OPTIONAL },
};
static const RadOperandSpec oKvStore[] = {
    { "k",            RAD_OPD_IN,    0 },
    { "v",            RAD_OPD_IN,    0 },
    { "slot_mapping", RAD_OPD_IN,    0 },
    { "kv_cache",     RAD_OPD_INOUT, 0 },
};

/* act_mul: verbatim libr4d (r4d_rows.cpp pActMul/oBinary). The gated MLP pair: act applied to
 * the gate operand, rounded to y's dtype, times the up operand. */
static const RadParamSpec pActMul[] = {
    { "M",    RAD_P_INT, RAD_REQUIRED },
    { "n",    RAD_P_INT, RAD_REQUIRED },
    { "act",  RAD_P_STR, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oBinary[] = {
    { "a", RAD_OPD_IN,  0 },
    { "b", RAD_OPD_IN,  0 },
    { "y", RAD_OPD_OUT, 0 },
};

/* cast: verbatim libr4d (r4d_rows.cpp pCast/oUnary). from/to are for the selector; the
 * tensors' own dtypes are what the kernel reads. */
static const RadParamSpec pCast[] = {
    { "M",    RAD_P_INT, RAD_REQUIRED },
    { "n",    RAD_P_INT, RAD_REQUIRED },
    { "from", RAD_P_STR, RAD_REQUIRED },
    { "to",   RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oUnary[] = {
    { "x", RAD_OPD_IN,  0 },
    { "y", RAD_OPD_OUT, 0 },
};

/* rope: verbatim libr4d (r4d_rows.cpp pRope/oRope). The yarn and llama3 frequency-scaling
 * parameters are declared (the schema must match libr4d's wherever both plugins load) but
 * this plugin's row refuses them; the bf16 text models that are libr3d's scope do not use them. */
static const RadParamSpec pRope[] = {
    { "M",          RAD_P_INT, RAD_REQUIRED },
    { "head_dim",   RAD_P_INT, RAD_REQUIRED },
    { "n_head",     RAD_P_INT, RAD_REQUIRED },
    { "n_head_kv",  RAD_P_INT, RAD_REQUIRED },
    { "theta",      RAD_P_F64, RAD_REQUIRED },
    { "scale",      RAD_P_F64, RAD_REQUIRED },
    { "mode",       RAD_P_STR, RAD_REQUIRED },
    { "rotary_dim", RAD_P_INT, RAD_OPTIONAL },
    { "sections",   RAD_P_STR, RAD_OPTIONAL },
    { "yarn_factor", RAD_P_F64, RAD_OPTIONAL },
    { "yarn_low", RAD_P_F64, RAD_OPTIONAL },
    { "yarn_high", RAD_P_F64, RAD_OPTIONAL },
    { "yarn_magnitude", RAD_P_F64, RAD_OPTIONAL },
    { "ladder",     RAD_P_STR, RAD_OPTIONAL },   /* the base ladder's width: rotary|head */
    /* Llama 3's frequency scaling (abi/rad_yarn.h, RadLlama3). */
    { "llama3_factor", RAD_P_F64, RAD_OPTIONAL },
    { "llama3_low",    RAD_P_F64, RAD_OPTIONAL },
    { "llama3_high",   RAD_P_F64, RAD_OPTIONAL },
    { "llama3_orig",   RAD_P_F64, RAD_OPTIONAL },
};
static const RadOperandSpec oRope[] = {
    { "qkv",       RAD_OPD_INOUT, 0 },
    { "positions", RAD_OPD_IN,    0 },
};

/* gemm_nt: verbatim libr4d (r4d_rows.cpp pGemmNt/oGemmNt). The quantised variants
 * (gemm_nt_q, _bias, gated) have no rows here: RDNA3 has no fp8 hardware and the int
 * grids are later tiers. */
static const RadParamSpec pGemmNt[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "N",     RAD_P_INT, RAD_REQUIRED },
    { "K",     RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oGemmNt[] = {
    { "a", RAD_OPD_IN,     0 },
    { "b", RAD_OPD_WEIGHT, 0 },
    { "y", RAD_OPD_OUT,    0 },
};

/* logits_gemm: verbatim libr4d (r4d_rows.cpp pLogits/oLogits). Same three operands in the
 * same order as gemm_nt, the extents spelled n_vocab / n_embd. */
static const RadParamSpec pLogits[] = {
    { "M",       RAD_P_INT, RAD_REQUIRED },
    { "n_vocab", RAD_P_INT, RAD_REQUIRED },
    { "n_embd",  RAD_P_INT, RAD_REQUIRED },
    { "dtype",   RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oLogits[] = {
    { "x",       RAD_OPD_IN,     0 },
    { "lm_head", RAD_OPD_WEIGHT, 0 },
    { "logits",  RAD_OPD_OUT,    0 },
};

/* attn_paged, the parameter and operand lists copied verbatim from libr4d's pAttnPaged /
 * oAttnPaged (r4d_rows.cpp) so co-load schema_same() passes. block_size is RAD_DERIVED --
 * read off the cache tensor, its constraint admits any value -- and the trailing
 * n_head / max_seqs exist for libr4d's split-KV scratch hook; ref ignores them and so does
 * our row, but both plugins must agree on the list. */
static const RadParamSpec pAttnPaged[] = {
    { "q_len",      RAD_P_INT, RAD_REQUIRED },
    { "head_dim",   RAD_P_INT, RAD_REQUIRED },
    { "gqa",        RAD_P_INT, RAD_REQUIRED },
    { "block_size", RAD_P_INT, RAD_DERIVED  },
    { "causal",     RAD_P_INT, RAD_REQUIRED },
    { "window",     RAD_P_INT, RAD_REQUIRED },
    { "q_dtype",    RAD_P_STR, RAD_REQUIRED },
    { "kv_dtype",   RAD_P_STR, RAD_REQUIRED },
    { "scale",      RAD_P_F64, RAD_OPTIONAL },
    { "max_ctx",    RAD_P_INT, RAD_OPTIONAL },
    { "n_head",     RAD_P_INT, RAD_OPTIONAL },
    { "max_seqs",   RAD_P_INT, RAD_OPTIONAL },
};
static const RadOperandSpec oAttnPaged[] = {
    { "q",           RAD_OPD_IN,  0 },
    { "kv_cache",    RAD_OPD_IN,  0 },
    { "block_table", RAD_OPD_IN,  0 },
    { "seqused",     RAD_OPD_IN,  0 },
    { "k_descale",   RAD_OPD_IN,  1 },
    { "v_descale",   RAD_OPD_IN,  1 },
    { "cu_seqlens",  RAD_OPD_IN,  1 },
    { "out",         RAD_OPD_OUT, 0 },
};

/* ---- sampling chain specs (verbatim from libr4d: pMVocab/pTopk/pMCand, oTemp/oMask/
 * oTopk/oArgmax/oNarrow/oPick). The engine's `params` operand is the RadSampleParams byte
 * blob (abi/rad_sample.h); n_vocab/n_cand name the leading dimension, vocab_off is the
 * shard origin and stays 0 in this single-rank scope. */
static const RadParamSpec pMVocab[] = {
    { "M",       RAD_P_INT, RAD_REQUIRED },
    { "n_vocab", RAD_P_INT, RAD_REQUIRED },
};
static const RadParamSpec pTopk[] = {
    { "M",         RAD_P_INT, RAD_REQUIRED },
    { "n_vocab",   RAD_P_INT, RAD_REQUIRED },
    { "vocab_off", RAD_P_INT, RAD_OPTIONAL },
};
static const RadParamSpec pMCand[] = {
    { "M",      RAD_P_INT, RAD_REQUIRED },
    { "n_cand", RAD_P_INT, RAD_REQUIRED },
};
/* ---- the history/summary stages (verbatim from libr4d: pPenal/oPenalties/oDry and
 * pLogprobs/oLogprobs). typical and xtc reuse pMCand/oNarrow -- same three operands,
 * narrowed in place. ---- */
static const RadParamSpec pPenal[] = {
    { "M",         RAD_P_INT, RAD_REQUIRED },
    { "n_vocab",   RAD_P_INT, RAD_REQUIRED },
    { "vocab_off", RAD_P_INT, RAD_OPTIONAL },
};
static const RadOperandSpec oPenalties[] = {
    { "logits",  RAD_OPD_INOUT, 0 },
    { "params",  RAD_OPD_IN,    0 },
    { "history", RAD_OPD_IN,    0 },
};
static const RadOperandSpec oDry[] = {
    { "logits",   RAD_OPD_INOUT, 0 },
    { "params",   RAD_OPD_IN,    0 },
    { "history",  RAD_OPD_IN,    0 },
    { "breakers", RAD_OPD_IN,    1 },
};
static const RadParamSpec pLogprobs[] = {
    { "M",         RAD_P_INT, RAD_REQUIRED },
    { "n_vocab",   RAD_P_INT, RAD_REQUIRED },
    { "k",         RAD_P_INT, RAD_REQUIRED },
    { "vocab_off", RAD_P_INT, RAD_OPTIONAL },
};
static const RadOperandSpec oLogprobs[] = {
    { "logits", RAD_OPD_IN,  0 },
    { "params", RAD_OPD_IN,  0 },
    { "stats",  RAD_OPD_OUT, 0 },
};
/* ---- band-op specs (verbatim from libr4d, the plugin that fixes these schemas). ---- */
static const RadParamSpec pAdd[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
    { "div",   RAD_P_F64, RAD_OPTIONAL },
};
/* oBinary (a, b, y) is already declared above for the act/cast batch -- identical table. */
static const RadParamSpec pMnDtype[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oGather[] = {
    { "x",   RAD_OPD_IN,  0 },
    { "idx", RAD_OPD_IN,  0 },
    { "y",   RAD_OPD_OUT, 0 },
};
static const RadParamSpec pQuantActFp8[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    { "group", RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oQuantActFp8[] = {
    { "x",     RAD_OPD_IN,  0 },
    { "q",     RAD_OPD_OUT, 0 },
    { "scale", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pGatedQuantFp8[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    { "group", RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
    { "act",   RAD_P_STR, RAD_OPTIONAL },
};
static const RadOperandSpec oGatedQuantFp8[] = {
    { "gate_up",  RAD_OPD_IN,  0 },
    { "q",        RAD_OPD_OUT, 0 },
    { "scale",    RAD_OPD_OUT, 0 },
    { "out_bf16", RAD_OPD_OUT, 1 },
};
static const RadOperandSpec oTemp[] = {
    { "logits", RAD_OPD_INOUT, 0 },
    { "params", RAD_OPD_IN,    0 },
};
static const RadOperandSpec oMask[] = {
    { "logits",  RAD_OPD_INOUT, 0 },
    { "params",  RAD_OPD_IN,    0 },
    { "bitmask", RAD_OPD_IN,    0 },
};
static const RadOperandSpec oTopk[] = {
    { "logits",   RAD_OPD_IN,  0 },
    { "params",   RAD_OPD_IN,  0 },
    { "cand_idx", RAD_OPD_OUT, 0 },
    { "cand_val", RAD_OPD_OUT, 0 },
    { "pairs",    RAD_OPD_OUT, 1 },
};
static const RadOperandSpec oArgmax[] = {
    { "logits", RAD_OPD_IN,  0 },
    { "params", RAD_OPD_IN,  0 },
    { "token",  RAD_OPD_OUT, 0 },
};
static const RadOperandSpec oNarrow[] = {
    { "cand_idx", RAD_OPD_INOUT, 0 },
    { "cand_val", RAD_OPD_INOUT, 0 },
    { "params",   RAD_OPD_IN,    0 },
};
static const RadOperandSpec oPick[] = {
    { "cand_idx", RAD_OPD_IN,  0 },
    { "cand_val", RAD_OPD_IN,  0 },
    { "params",   RAD_OPD_IN,    0 },
    { "token",    RAD_OPD_OUT, 0 },
};

/* silu_mul: pMnDtype is the same table gather_rows declares (verbatim from libr4d's, as
 * is oGateUp) -- the loader's schema check compares the fixed grammar, not pointers. */
static const RadOperandSpec oGateUp[] = {
    { "gate_up", RAD_OPD_IN,  0 },
    { "y",       RAD_OPD_OUT, 0 },
};
/* quant_act_i8: params identical to quant_act_fp8's; the optional fourth output selects
 * the _asum form (verbatim libr4d oQuantAct). */
static const RadOperandSpec oQuantAct[] = {
    { "x",     RAD_OPD_IN,  0 },
    { "q",     RAD_OPD_OUT, 0 },
    { "scale", RAD_OPD_OUT, 0 },
    { "asum",  RAD_OPD_OUT, 1 },
};
/* row_topk (verbatim libr4d pRowTopk / oRowTopk). */
static const RadParamSpec pRowTopk[] = {
    { "M",         RAD_P_INT, RAD_REQUIRED },
    { "N",         RAD_P_INT, RAD_REQUIRED },
    { "R",         RAD_P_INT, RAD_REQUIRED },
    { "vocab_off", RAD_P_INT, RAD_OPTIONAL },
    { "dtype",     RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oRowTopk[] = {
    { "x",     RAD_OPD_IN,  0 },
    { "idx",   RAD_OPD_OUT, 0 },
    { "val",   RAD_OPD_OUT, 0 },
    { "pairs", RAD_OPD_OUT, 1 },
};

static const RadOpSchema kSchemas[] = {
    { "rmsnorm", pRmsNorm, (int)(sizeof(pRmsNorm)/sizeof(pRmsNorm[0])),
      oRmsNorm,  (int)(sizeof(oRmsNorm)/sizeof(oRmsNorm[0])),
      "y = x * rsqrt(mean(x^2) + eps) * (w + wadd). Declared verbatim from libr4d so both "
      "plugins agree wherever they co-load." },
    { "embed_lookup", pEmbedM, (int)(sizeof(pEmbedM)/sizeof(pEmbedM[0])),
      oEmbed,  (int)(sizeof(oEmbed)/sizeof(oEmbed[0])),
      "the embedding gather. A negative token id writes a zero row, and so does one outside "
      "[vocab_offset, vocab_offset + n_vocab) -- which is what makes the table shardable "
      "across the vocabulary, the symmetry lm_head already had. Ids stay GLOBAL; summing the "
      "ranks reconstructs x, so the caller follows this with an exact all_reduce." },
    { "kv_store", pKvStore, (int)(sizeof(pKvStore)/sizeof(pKvStore[0])),
      oKvStore,  (int)(sizeof(oKvStore)/sizeof(oKvStore[0])),
      "scatter k and v into kv_cache[n_blocks, kv_heads, block_size, 2*head_dim] -- K at "
      "[.., d] and V at [.., head_dim + d]. A negative slot stores nothing." },
    { "act_mul", pActMul, (int)(sizeof(pActMul)/sizeof(pActMul[0])),
      oBinary, (int)(sizeof(oBinary)/sizeof(oBinary[0])),
      "y = act(a) * b, the activation (silu, gelu, gelu_tanh) rounded to y's dtype before "
      "the product: silu_mul with its two halves in two operands." },
    { "cast", pCast, (int)(sizeof(pCast)/sizeof(pCast[0])),
      oUnary, (int)(sizeof(oUnary)/sizeof(oUnary[0])),
      "y = x, converted. `from` and `to` are for the selector; the bytes are described by the "
      "tensors' own dtypes and those are what the kernel reads." },
    { "rope", pRope, (int)(sizeof(pRope)/sizeof(pRope[0])),
      oRope,  (int)(sizeof(oRope)/sizeof(oRope[0])),
      "rotary embedding in place over the Q and K heads of a fused projection; the V heads "
      "that follow are untouched. The multi-component modes (`mrope`, `imrope`, `axial`) "
      "take [C, M] component-major positions and `sections`; with [M] positions every "
      "component is that one, so mrope and imrope over a text batch are neox exactly." },
    { "gemm_nt", pGemmNt, (int)(sizeof(pGemmNt)/sizeof(pGemmNt[0])),
      oGemmNt, (int)(sizeof(oGemmNt)/sizeof(oGemmNt[0])),
      "Y[M,N] = A[M,K] @ B[N,K]^T. B is stored transposed (the layout transformer weights "
      "already have), so both operands are k-contiguous and the inner loop is a dot product "
      "over k." },
    { "logits_gemm", pLogits, (int)(sizeof(pLogits)/sizeof(pLogits[0])),
      oLogits, (int)(sizeof(oLogits)/sizeof(oLogits[0])),
      "the lm_head projection. gemm_nt's arithmetic under the vocabulary edge's parameter "
      "spelling; M is one row per SAMPLED position, `gather_rows` having run first." },
    { "attn_paged", pAttnPaged, (int)(sizeof(pAttnPaged)/sizeof(pAttnPaged[0])),
      oAttnPaged, (int)(sizeof(oAttnPaged)/sizeof(oAttnPaged[0])),
      "Paged attention over kv_cache[num_blocks, kv_heads, block_size, 2*head_dim] (K then "
      "V per slot), f32 softmax with the max subtracted, causal and sliding-window masking, "
      "q_len >= 1 so one kernel serves decode, speculative verify and prefill. seqused is "
      "the TOTAL key count including this step's query rows. k_descale / v_descale are "
      "optional and apply to an fp8 cache. `scale` is not in docs/OPS.md and defaults to "
      "1/sqrt(head_dim). Declared verbatim from libr4d so both plugins agree wherever they "
      "co-load." },

      /* ---- the sampling chain (classic stages only; see r3d_sample.hip's scope note). ---- */
      /* Param/operand tables verbatim from libr4d (pMVocab/pTopk/pMCand and the o* tables),
      * so the loader's schema-duplication check passes wherever both plugins co-load. */
      { "sample_temp", pMVocab, (int)(sizeof(pMVocab)/sizeof(pMVocab[0])),
       oTemp,   (int)(sizeof(oTemp)/sizeof(oTemp[0])),
       "per-request temperature. temp <= 0 leaves the row untouched; greedy is sample_argmax." },
      { "sample_mask", pMVocab, (int)(sizeof(pMVocab)/sizeof(pMVocab[0])),
       oMask,   (int)(sizeof(oMask)/sizeof(oMask[0])),
       "A SET BIT MEANS ALLOWED. Word w bit b covers token 32w + b; a cleared bit sets "
       "-infinity. The row's mask_row selects the plane row, or -1 when unconstrained." },
      { "sample_topk", pTopk, (int)(sizeof(pTopk)/sizeof(pTopk[0])),
       oTopk,   (int)(sizeof(oTopk)/sizeof(oTopk[0])),
       "the exact top-k candidate set, descending, ties to the lower token id. The width is "
       "the buffer's; a row's own top_k may be narrower and leaves holes." },
      { "sample_argmax", pMVocab, (int)(sizeof(pMVocab)/sizeof(pMVocab[0])),
       oArgmax, (int)(sizeof(oArgmax)/sizeof(oArgmax[0])),
       "greedy, ties to the lower token id." },
      { "sample_topp", pMCand, (int)(sizeof(pMCand)/sizeof(pMCand[0])),
       oNarrow, (int)(sizeof(oNarrow)/sizeof(oNarrow[0])),
       "nucleus: keep the shortest prefix whose cumulative probability reaches top_p, always "
       "at least one. The candidates are assumed descending, as sample_topk leaves them." },
      { "sample_minp", pMCand, (int)(sizeof(pMCand)/sizeof(pMCand[0])),
       oNarrow, (int)(sizeof(oNarrow)/sizeof(oNarrow[0])),
       "keep candidates whose probability is at least min_p times the most likely one's." },
      { "sample_pick", pMCand, (int)(sizeof(pMCand)/sizeof(pMCand[0])),
        oPick,   (int)(sizeof(oPick)/sizeof(oPick[0])),
        "the inverse-CDF draw over the candidate set, from the specified splitmix64 stream." },
      { "sample_penalties", pPenal, (int)(sizeof(pPenal)/sizeof(pPenal[0])),
        oPenalties, (int)(sizeof(oPenalties)/sizeof(oPenalties[0])),
        "repetition, frequency and presence penalties over the logits row, from the row's "
        "history window: distinct tokens only, the repetition divides above zero and "
        "multiplies at or below." },
      { "sample_dry", pPenal, (int)(sizeof(pPenal)/sizeof(pPenal[0])),
        oDry, (int)(sizeof(oDry)/sizeof(oDry[0])),
        "the DRY repetition-suffix penalty: the longest suffix of the window that repeats "
        "before each continuation, multiplier times base^(length - allowed), capped at the "
        "float overflow boundary. An optional breaker plane exempts positions." },
      { "sample_typical", pMCand, (int)(sizeof(pMCand)/sizeof(pMCand[0])),
        oNarrow, (int)(sizeof(oNarrow)/sizeof(oNarrow[0])),
        "locally typical narrowing: keep the candidates whose |surprisal - entropy| mass "
        "at-or-below their distance reaches typical_p; every candidate tied at the cut "
        "survives together." },
      { "sample_xtc", pMCand, (int)(sizeof(pMCand)/sizeof(pMCand[0])),
        oNarrow, (int)(sizeof(oNarrow)/sizeof(oNarrow[0])),
        "exclude-top-changes: with the row's own coin, drop every candidate at or above "
        "xtc_threshold except the last one; fires only when two or more are above." },
      { "sample_logprobs", pLogprobs, (int)(sizeof(pLogprobs)/sizeof(pLogprobs[0])),
        oLogprobs, (int)(sizeof(oLogprobs)/sizeof(oLogprobs[0])),
        "the raw distribution's summary: max, sum of exp, and the k largest (id, logit) "
        "pairs over the UNSOFTMAXED row; reads nothing the other stages wrote." },

      /* ---- the band ops that gate the quantising bf16 trunk (see r3d_band.hip). ---- */
      { "add", pAdd, (int)(sizeof(pAdd)/sizeof(pAdd[0])),
        oBinary, (int)(sizeof(oBinary)/sizeof(oBinary[0])),
        "y = a + b. `b` carrying exactly n elements against an output of M rows broadcasts "
        "along rows, and nothing else broadcasts at all. With `div`, y = rd(rd(a + b) / div), "
        "rd y's dtype." },
      { "gather_rows", pMnDtype, (int)(sizeof(pMnDtype)/sizeof(pMnDtype[0])),
        oGather, (int)(sizeof(oGather)/sizeof(oGather[0])),
        "y[i, :] = x[idx[i], :]. A negative index writes a zero row -- the padding convention a "
        "step with fewer sampled rows than the buffer's extent leaves behind." },
      { "gated_quant_fp8", pGatedQuantFp8, (int)(sizeof(pGatedQuantFp8)/sizeof(pGatedQuantFp8[0])),
        oGatedQuantFp8, (int)(sizeof(oGatedQuantFp8)/sizeof(oGatedQuantFp8[0])),
        "silu(a)*b or sigmoid(a)*b, then the same quantiser, in one pass." },
      { "quant_act_fp8", pQuantActFp8, (int)(sizeof(pQuantActFp8)/sizeof(pQuantActFp8[0])),
        oQuantActFp8, (int)(sizeof(oQuantActFp8)/sizeof(oQuantActFp8[0])),
        "per-(row, `group`-column) symmetric E4M3 quantisation of a token-major activation, in "
        "PLAIN ROW-MAJOR order rather than quant_act_i8's fragment order. The scale is "
        "amax/448 -- a DEQUANT multiplier, the same sense as the checkpoint's "
        "weight_scale_inv, so the fp8 GEMMs multiply both scales and never divide." },
      { "silu_mul", pMnDtype, (int)(sizeof(pMnDtype)/sizeof(pMnDtype[0])),
        oGateUp, (int)(sizeof(oGateUp)/sizeof(oGateUp[0])),
        "y[.., i] = silu(gate_up[.., i]) * gate_up[.., n + i]. The GATE IS THE FIRST HALF, "
        "which is the order rad-convert fuses gate_proj and up_proj in." },
      { "quant_act_i8", pQuantActFp8, (int)(sizeof(pQuantActFp8)/sizeof(pQuantActFp8[0])),
        oQuantAct, (int)(sizeof(oQuantAct)/sizeof(oQuantAct[0])),
        "per-row symmetric int8 quantisation, written in the byte order the r4d GEMMs' A "
        "fragment reads. A third output selects the form that also emits the per-group row "
        "sums an asymmetric weight grid needs." },
      { "row_topk", pRowTopk, (int)(sizeof(pRowTopk)/sizeof(pRowTopk[0])),
        oRowTopk, (int)(sizeof(oRowTopk)/sizeof(oRowTopk[0])),
        "exact per-row top-R, descending, ties to the lower column. `vocab_off` shifts every "
        "column reported so a rank's slice reports global ids, and `pairs` is the [M, R, 2] "
        "(int32 id, float value) plane an all_gather moves; a slot the row could not fill "
        "reports id -1 there and column 0 in `idx`, which is the unsharded contract unchanged." },
      };

/* The rmsnorm row. Field order is RadKernelInfo's positional form (rad_abi.h): after the prose,
 * constraints and (absent) tunables come init, fini, launch, scratch, layout, relayout, and
 * opd_shape -- null because a cold-calling tool takes operand shapes from libref's for the op.
 * `replaces` stays zero-initialised: this row is the unfused op, not a fusion. */

static const RadConstraint cBf16Row[] = {
    { "dtype", RAD_C_IN, 0, "bf16" },
};

/* logits_gemm bf16: verbatim libr4d cLogitsGemm = { C_DIV("n_embd", 8), C_IN("dtype", "bf16") }. */
static const RadConstraint cLogitsGemm[] = {
    { "n_embd", RAD_C_DIV, 8,    nullptr },
    { "dtype",  RAD_C_IN,  0,    "bf16" },
};

/* kv_store constraints, the positional form of libr4d's cKvStore = { C_GE("block_size", 1),
 * C_IN("kv_dtype", "bf16") }. */
static const RadConstraint cKvStore[] = {
    { "block_size", RAD_C_GE, 1, nullptr },
    { "kv_dtype",   RAD_C_IN, 0, "bf16" },
};

/* libr3d is the only device library on RDNA3, so no in-plugin arbitration; the value only
 * needs to sit above a future fallback row. */
enum { R3D_PRIO_ONLY = 1000 };

/* act_mul: the bf16 gated MLP, all four oracle activations this kernel ports. */
static const RadConstraint cActMul[] = {
    { "dtype", RAD_C_IN, 0, "bf16" },
    { "act",   RAD_C_IN, 0, "silu gelu gelu_tanh sigmoid" },
};

/* The cast pairs this plugin serves, one row each (libr4d spells the same four; the f16
 * narrowing is not yet ported and its pair simply has no row here). */
static const RadConstraint cCastBf16[] = {
    { "from", RAD_C_IN, 0, "bf16" }, { "to", RAD_C_IN, 0, "bf16" },
};
static const RadConstraint cCastWiden[] = {
    { "from", RAD_C_IN, 0, "bf16" }, { "to", RAD_C_IN, 0, "f32" },
};
static const RadConstraint cCastNarrow[] = {
    { "from", RAD_C_IN, 0, "f32" }, { "to", RAD_C_IN, 0, "bf16" },
};

/* attn_paged: the row admits what the kernel serves. head_dim is a thread count here, so
 * the entry point refuses > 1024; the constraint stays open above that and a query for a
 * bigger head resolves past the row to nothing -- an honest miss. f32 q is served by the
 * kernel but not advertised: the model path is bf16, and a constraint the fixtures never
 * probe is a constraint never checked. */
static const RadConstraint cAttnPaged[] = {
    { "head_dim", RAD_C_GE, 1, nullptr },
    { "q_dtype",  RAD_C_IN, 0, "bf16" },
    { "kv_dtype", RAD_C_IN, 0, "bf16 f32 fp8_e4m3" },
};

/* rope: neox and gptj over [M] positions -- a subset of libr4d's five modes. A query for
 * mrope/imrope/axial resolves to no device row on this card, which is an honest failure at
 * issue, not an approximation. */
static const RadConstraint cRopeRow[] = {
    { "head_dim", RAD_C_GE, 1, nullptr },
    { "mode",     RAD_C_IN, 0, "neox gptj" },
};

/* Sampler row constraints, mirroring libr4d's cSampleVoc / cSampleCand. */
static const RadConstraint cSampleVoc[]  = { { "n_vocab", RAD_C_GE, 1, nullptr } };
static const RadConstraint cSampleCand[] = { { "n_cand", RAD_C_GE, 1, nullptr },
                                             { "n_cand", RAD_C_LE, 4096, nullptr } };

/* Band-op constraints: the widths the kernels stride, and the activation dtypes
 * r3d_band.hip actually reads (f32/bf16) -- an f16 band resolves to the reference
 * instead of being refused at launch. */
static const RadConstraint cBandMn[] = {
    { "n",     RAD_C_GE, 1, nullptr },
    { "dtype", RAD_C_IN, 0, "f32 bf16" },
};

extern "C" int r3d_rad_rmsnorm(const RadArgs*, RadStream);
extern "C" int r3d_rad_embed_lookup(const RadArgs*, RadStream);
extern "C" int r3d_rad_kv_store(const RadArgs*, RadStream);
extern "C" int r3d_rad_act_mul(const RadArgs*, RadStream);
extern "C" int r3d_rad_cast(const RadArgs*, RadStream);
extern "C" int r3d_rad_rope(const RadArgs*, RadStream);
extern "C" int r3d_rad_gemm_nt(const RadArgs*, RadStream);
extern "C" int r3d_rad_logits_gemm(const RadArgs*, RadStream);
extern "C" int r3d_rad_attn_paged(const RadArgs*, RadStream);
extern "C" int r3d_rad_sample_temp(const RadArgs*, RadStream);
extern "C" int r3d_rad_sample_mask(const RadArgs*, RadStream);
extern "C" int r3d_rad_sample_argmax(const RadArgs*, RadStream);
extern "C" int r3d_rad_sample_topk(const RadArgs*, RadStream);
extern "C" int r3d_rad_sample_topp(const RadArgs*, RadStream);
extern "C" int r3d_rad_sample_minp(const RadArgs*, RadStream);
extern "C" int r3d_rad_sample_pick(const RadArgs*, RadStream);
extern "C" int r3d_rad_sample_penalties(const RadArgs*, RadStream);
extern "C" int r3d_rad_sample_dry(const RadArgs*, RadStream);
extern "C" int r3d_rad_sample_typical(const RadArgs*, RadStream);
extern "C" int r3d_rad_sample_xtc(const RadArgs*, RadStream);
extern "C" int r3d_rad_sample_logprobs(const RadArgs*, RadStream);
extern "C" int r3d_rad_quant_act_fp8(const RadArgs*, RadStream);
extern "C" int r3d_rad_gated_quant_fp8(const RadArgs*, RadStream);
extern "C" int r3d_rad_add(const RadArgs*, RadStream);
extern "C" int r3d_rad_gather_rows(const RadArgs*, RadStream);
extern "C" int r3d_rad_silu_mul(const RadArgs*, RadStream);
extern "C" int r3d_rad_quant_act_i8(const RadArgs*, RadStream);
extern "C" int r3d_rad_row_topk(const RadArgs*, RadStream);

/* Constraints for the r3d_more rows, in libr4d's shape so a query that resolves there
 * resolves here identically: silu_mul is the plain bf16 row; quant_act_i8 walks the
 * sixteen-byte WMMA step so `n DIV 16`; row_topk admits N a multiple of 8 and R <= 32. */
static const RadConstraint cQuantActI8[] = {
    { "n",     RAD_C_DIV, 16,   nullptr },
    { "dtype", RAD_C_IN,  0,    "bf16" },
};
static const RadConstraint cRowTopK[] = {
    { "N",     RAD_C_DIV, 8,    nullptr },
    { "M",     RAD_C_GE,  1,    nullptr },
    { "R",     RAD_C_GE,  1,    nullptr },
    { "R",     RAD_C_LE,  32,   nullptr },
    { "dtype", RAD_C_IN,  0,    "bf16" },
};

static const RadKernelInfo kRows[] = {

{ "rmsnorm_bf16", "rmsnorm", "norm",
  "root-mean-square norm with a gain, two passes over the row",
  "any n; the sum of squares accumulates in f32 and only the store narrows",
  "bf16 x and y, bf16 or f32 gain, fp32 reduction",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cBf16Row, 1, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_rmsnorm },

{ "embed_lookup_bf16", "embed_lookup", "vocab",
  "embedding gather, one workgroup a token",
  "any n_embd and n_vocab; a negative or out-of-range id writes a zero row",
  "i32 tokens, bf16 table and output",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cBf16Row, 1, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_embed_lookup },

{ "kv_store_bf16", "kv_store", "attn",
  "scatter k and v into the paged cache, one workgroup a token",
  "cache exactly [n_blocks, kv_heads, block_size, 2*head_dim]; a negative or out-of-range slot "
  "stores nothing",
  "bf16 k / v / cache, i32 slot mapping",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cKvStore, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_kv_store },

{ "act_mul_bf16", "act_mul", "elem",
  "gated activation over two operands: act(a) rounded, times b",
  "any n; a, b and y each at their own row pitch, so both may be column slices of one buffer",
  "bf16 in and out, f32 arithmetic, double-width silu / erf / tanh like the oracle",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cActMul, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_act_mul },

{ "cast_bf16_bf16", "cast", "elem",
  "y = x, bf16 to bf16 -- a strided row copy",
  "any M and n; both row pitches are read off the operands, so a column slice of a wider "
  "buffer is expressible",
  "bf16 in, bf16 out, byte-for-byte",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cCastBf16, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_cast },

{ "cast_bf16_f32", "cast", "elem",
  "y = x, bf16 widened to f32",
  "any M and n; both row pitches are read off the operands",
  "bf16 in, f32 out, exact",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cCastWiden, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_cast },

{ "cast_f32_bf16", "cast", "elem",
  "y = x, f32 narrowed to bf16",
  "any M and n; both row pitches are read off the operands",
  "f32 in, bf16 out, rounded to nearest even",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cCastNarrow, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_cast },

{ "rope_bf16", "rope", "attn",
  "rotary embedding in place, one thread a rotated pair, the ladder built once per token",
  "any head_dim; rotary_dim even and no wider than head_dim; neox and gptj pairings over "
  "[M] positions (the multi-component modes and yarn/llama3 scaling have no row here)",
  "bf16 qkv, i32 positions, fp32 angles through the accurate sincosf",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cRopeRow, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_rope },

{ "gemm_nt_bf16", "gemm_nt", "gemm",
  "dense bf16 GEMM, 64x64 output tile per workgroup, K in 16-wide LDS slices",
  "any M, N and K (to 2^27); ragged tiles load as zero and are never stored; operand row "
  "pitches follow the trailing-dimension fold, so a rank-3 qkv output is writable",
  "bf16 A and B, bf16 or f32 Y, f32 accumulation",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cBf16Row, 1, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_gemm_nt },

{ "logits_gemm_bf16", "logits_gemm", "vocab",
  "the lm_head projection: same tile as gemm_nt_bf16, the vocabulary edge's parameter "
  "spelling; logits go out F32",
  "any M and any vocabulary; n_embd a multiple of 8",
  "bf16 x and lm_head, F32 LOGITS OUT, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cLogitsGemm, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_logits_gemm },

{ "attn_paged_bf16", "attn_paged", "attn",
  "paged causal / sliding-window attention, one workgroup per (query row, head), any "
  "query length, varlen, split-free (no scratch)",
  "any head_dim to 1024, any gqa, any block size, window and causal masking, ragged "
  "batches via cu_seqlens, unmapped (negative) block table entries skipped",
  "bf16 query, bf16 / f32 / fp8-e4m3 kv cache with optional per-(seq,head) descales, "
  "bf16 or f32 out, f32 softmax with the max subtracted",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cAttnPaged, 3, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_attn_paged },

/* ---- the sampling chain. The candidate-width cap on the cand-plane rows mirrors
 * libr4d's: the engine's declare-time bound is the buffer width, and 4096 covers it.
 * sample_topk here is k parallel argmax rounds under the (v desc, id asc) order --
 * exact, deterministic, no radix sort; the vocab widths the fixtures record make the
 * re-reads L2-resident, and a correctness-first pass is the point of this library. */

{ "sample_temp_f32", "sample_temp", "sample",
  "per-request temperature over the logits row",
  "any vocabulary; a row with temp <= 0 or exactly 1 is skipped whole",
  "f32 logits in place, per-row params",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cSampleVoc, 1, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_sample_temp },

{ "sample_mask_f32", "sample_mask", "sample",
  "the grammar bitmask -- a cleared bit sets -infinity",
  "any vocabulary; the plane is [rows, ceil(n_vocab/32)], addressed by the row's mask_row",
  "f32 logits in place, u32 bitmask, per-row params",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cSampleVoc, 1, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_sample_mask },

{ "sample_argmax_f32", "sample_argmax", "sample",
  "greedy, ties to the lower token id",
  "any vocabulary; the block reduction uses the same (value desc, id asc) order as top-k, "
  "so greedy and top-1 agree",
  "f32 logits, i32 token",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cSampleVoc, 1, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_sample_argmax },

{ "sample_topk_f32", "sample_topk", "sample",
  "the exact top-k candidate set, descending, ties to the lower token id",
  "any vocabulary; n_cand <= 4096; k rounds of block argmax under the (value desc, id asc) "
  "order reproduce the reference selection exactly, including the tie window after the "
  "previous pick; the optional vocab-parallel pairs plane is packed when present",
  "f32 logits, i32 candidate ids, f32 candidate LOGITS (not probabilities)",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cSampleVoc, 1, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_sample_topk },

{ "sample_topp_f32", "sample_topp", "sample",
  "nucleus narrowing in place",
  "n_cand <= 4096; the prefix sum is SEQUENTIAL in f32 in candidate order -- a parallel "
  "prefix sums in a different order and a candidate at the cut would survive in one and "
  "not the other",
  "i32 candidate ids and f32 candidate logits, both in place",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cSampleCand, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_sample_topp },

{ "sample_minp_f32", "sample_minp", "sample",
  "min-p narrowing in place",
  "n_cand <= 4096; exp ratios against the max in candidate order, the max always survives",
  "i32 candidate ids and f32 candidate logits, both in place",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cSampleCand, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_sample_minp },

{ "sample_pick_f32", "sample_pick", "sample",
  "the inverse-CDF draw over the candidate set, from the spec splitmix64 stream",
  "n_cand <= 4096; one thread per row accumulates in candidate order so the draw "
  "reproduces the reference token for token; a fall-through takes the last candidate "
  "ABOVE -infinity, never a masked tail entry",
  "i32 candidate ids, f32 candidate logits, i32 token out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cSampleCand, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_sample_pick },

{ "sample_penalties_f32", "sample_penalties", "sample",
  "repetition / frequency / presence penalties from the row's history window",
  "any vocabulary; the walk is per DISTINCT token of the window, so the cost is the "
  "window, not the vocabulary; the i32 history plane must be contiguous",
  "f32 logits in place, i32 history plane, per-row params",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cSampleVoc, 1, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_sample_penalties },

{ "sample_dry_f32", "sample_dry", "sample",
  "the DRY repetition-suffix penalty, the naive longest-suffix scan of the oracle",
  "the window is bounded by what fits LDS (hmax <= 6144); the penalty is the "
  "repeated-multiplication form with the FLT_MAX/base stop, computed identically by every "
  "implementation",
  "f32 logits in place, i32 history plane, optional i32 breaker plane, per-row params",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cSampleVoc, 1, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_sample_dry },

{ "sample_typical_f32", "sample_typical", "sample",
  "locally typical narrowing in place",
  "n_cand <= 4096; the entropy and the cut scan are SEQUENTIAL in candidate order on one "
  "thread -- the entropy sum and the mass scan are order-sensitive sums, and the tie rule "
  "(everything at the cut survives) is the llama.cpp one the oracle documents",
  "i32 candidate ids and f32 candidate logits, both in place",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cSampleCand, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_sample_typical },

{ "sample_xtc_f32", "sample_xtc", "sample",
  "exclude-top-changes narrowing in place, from the row's own XTC coin",
  "n_cand <= 4096; the coin is the spec stream at RAD_SAMPLE_KEY_XTC, so the firing "
  "decision is reproducible; ref fires on `>=` and a draw exactly on the boundary fires",
  "i32 candidate ids and f32 candidate logits, both in place",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cSampleCand, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_sample_xtc },

{ "sample_logprobs_f32", "sample_logprobs", "sample",
  "the raw distribution's summary: max, double sum of exp, and the k largest pairs",
  "any vocabulary; k rounds of the block argmax under the one total order reproduce the "
  "reference's insertion-sorted top-k; a vocabulary narrower than k leaves the remaining "
  "slots as (-1, -inf) holes",
  "f32 logits (read, not written), f32 stats row [M, 2 + 2k]",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cSampleVoc, 1, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_sample_logprobs },

/* ---- the band ops. Until these exist, a bf16 trunk that quantises its MLP activations
 * cannot START on this card at all: quant_act_fp8 / gated_quant_fp8 / add / gather_rows
 * carry no weights, so the planner cannot place them on a host kernel and the graph is
 * unsupported at resolve. fp8 is storage here, not arithmetic -- no E4M3 hardware is
 * touched, only the ABI encoder. */

{ "quant_act_fp8_act", "quant_act_fp8", "band",
  "per-group symmetric E4M3 quantisation of an activation, plain row-major",
  "f32/bf16 inputs; the scale is amax/448, a dequant multiplier; group clamped to n",
  "activation in, f8e4m3 codes and f32 group scales out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cBandMn, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_quant_act_fp8 },

{ "gated_quant_fp8_act", "gated_quant_fp8", "band",
  "silu(gate)*up (or sigmoid(gate)*up) fused with the same quantiser",
  "f32/bf16 gate_up [M, 2n]; the gated value narrows to bf16 once and the quantiser sees "
  "the widened narrow; the gate is recomputed per pass rather than staged, deterministically",
  "f32/bf16 gate_up in, f8e4m3 codes, f32 scales, optional bf16/f32 gated output",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cBandMn, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_gated_quant_fp8 },

{ "add_act", "add", "band",
  "elementwise add with the double-rounded div form",
  "f32/bf16 operands; b broadcasts along rows only at exactly n elements; with div, "
  "y = rd(rd(a+b)/div)",
  "f32/bf16 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cBandMn, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_add },

{ "gather_rows_act", "gather_rows", "band",
  "row gather with the zero-row padding convention",
  "f32/bf16 rows, i32 indices; a negative index writes a zero row; an index >= the "
  "source rows ALSO writes zeros rather than failing the launch (a device kernel cannot "
  "pre-validate like the host reference; documented deviation)",
  "f32/bf16 x and y, i32 idx",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cBandMn, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_gather_rows },

/* ---- the three bands MiniCPM5-2B's graph names at resolve (see r3d_more.hip). The
 * engine refused to START without them: weightless device bands have no host stand-in. */

{ "silu_mul_bf16", "silu_mul", "elem",
  "silu-gated product over a fused [M, 2n] gate_up, one f32 expression chain per element",
  "any n; the gate is the first half",
  "bf16 in and out, fp32 arithmetic, double-width sigmoid like the oracle",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cBf16Row, 1, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_silu_mul },

{ "quant_act_i8", "quant_act_i8", "quant",
  "per-row symmetric int8 quantisation of a bf16 activation, written in the byte order "
  "the r4d GEMMs' A fragment reads; a third output selects the form that also emits the "
  "per-group row sums",
  "n a multiple of 16; one workgroup per row",
  "bf16 in, int8 out with an f32 per-row scale",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cQuantActI8, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_quant_act_i8 },

{ "rowtopk_bf16", "row_topk", "select",
  "exact per-row top-R of a bf16 matrix, descending, ties to the lower column; R rounds "
  "of the (value desc, id asc) block argmax reproduce topk_next's window exactly",
  "any M, N a multiple of 8, R <= 32",
  "bf16 in, f32 values + i32 columns out, optional f32 pairs plane",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cRowTopK, 5, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_row_topk },

};

int rad_kernel_schema_count(void) { return (int)(sizeof(kSchemas) / sizeof(kSchemas[0])); }
const RadOpSchema* rad_kernel_schema_at(int i) {
    return (i >= 0 && i < rad_kernel_schema_count()) ? &kSchemas[i] : nullptr;
}

int rad_kernel_count(void) { return (int)(sizeof(kRows) / sizeof(kRows[0])); }
const RadKernelInfo* rad_kernel_at(int i) {
    return (i >= 0 && i < rad_kernel_count()) ? &kRows[i] : nullptr;
}

} /* extern "C" */
