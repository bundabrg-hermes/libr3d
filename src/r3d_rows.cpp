/* r3d_rows.cpp -- libr3d's declaration: plugin identity and the kernel row table.
 *
 * Pure host C++ including only the ABI, so it compiles without ROCm, exactly like libr4d's
 * r4d_rows.cpp. This skeleton exports the full kernel-plugin contract with an empty table:
 * the loader takes it (ABI matches), finds no device rows for gfx11, and selection behaves as
 * if libr3d were absent. Real rows land here as kernels are implemented; docs/COVERAGE.md
 * tracks the order.
 */

#include "rad_abi.h"
#include <cstdlib>   // getenv, for the R3D_NO_GDN registration filter
#include <cstring>   // strcmp, same

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

/* embed_lookup_q: param/operand tables verbatim from libr4d (r4d_rows.cpp pEmbed/oEmbedQ),
 * which are verbatim libref's (ref_registry.cpp) -- the loader refuses a kernel plugin whose
 * positional schema disagrees with the op's first declarant WHOLE, so this list must match
 * entry for entry: pEmbed is pEmbedM without `mult` (the quantised table has its scale
 * operand instead of a row factor), and oEmbedQ adds the optional `scale` weight -- one
 * entry being the n-gram's whole-table scale, more entries the per-tile plane of a
 * block-fp8 table -- and the optional `ahead` ids of a host reader that fetches rows. */
static const RadParamSpec pEmbedQ[] = {
    { "M",       RAD_P_INT, RAD_REQUIRED },
    { "n_embd",  RAD_P_INT, RAD_REQUIRED },
    { "n_vocab", RAD_P_INT, RAD_REQUIRED },
    { "dtype",   RAD_P_STR, RAD_REQUIRED },
    { "vocab_offset", RAD_P_INT, RAD_OPTIONAL },
    { "wscale",  RAD_P_F64, RAD_OPTIONAL },
};
static const RadOperandSpec oEmbedQ[] = {
    { "tokens", RAD_OPD_IN,     0 },
    { "wte",    RAD_OPD_WEIGHT, 0 },
    { "scale",  RAD_OPD_WEIGHT, 1 },
    { "x",      RAD_OPD_OUT,    0 },
    { "ahead",  RAD_OPD_IN,     1 },
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

/* gemm_nt: verbatim libr4d (r4d_rows.cpp pGemmNt/oGemmNt). The quantised gemm_nt_q is
 * declared next; its mxfp4a8 encoding has a row (r3d_gemm_mxfp4.hip, bf16 boundary
 * conversion -- RDNA3 has no fp8/fp4 hardware), the int grids are later tiers, and the
 * _bias/gated variants have no rows yet. */
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

/* gemm_nt_q: verbatim libr4d (r4d_rows.cpp pGemmNtQ/oGemmNtQ) so schema_same() passes
 * when the plugins co-load. a_sum/b_ref are the seventh/sixth operand tail RadArgs
 * cannot carry as weight aux (the ABI gap libr4d's comment spells out); the mxfp4a8
 * entry refuses a_sum, and accepts-and-ignores b_ref exactly as both host libraries. */
static const RadParamSpec pGemmNtQ[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "N",     RAD_P_INT, RAD_REQUIRED },
    { "K",     RAD_P_INT, RAD_REQUIRED },
    { "group", RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
    { "pb1",   RAD_P_INT, RAD_OPTIONAL },
    { "pb2",   RAD_P_INT, RAD_OPTIONAL },
    /* v1.11: libr4d's K-split hint (docs/OPS.md). Accept and ignore -- libr3d never
     * splits K, and both plugins must agree on the list or the engine refuses them. */
    { "split_n", RAD_P_INT, RAD_OPTIONAL },
};
static const RadOperandSpec oGemmNtQ[] = {
    { "a",       RAD_OPD_IN,     0 },
    { "a_scale", RAD_OPD_IN,     1 },
    { "b",       RAD_OPD_WEIGHT, 0 },
    { "b_scale", RAD_OPD_WEIGHT, 0 },
    { "y",       RAD_OPD_OUT,    0 },
    { "a_sum",   RAD_OPD_IN,     1 },
    { "b_ref",   RAD_OPD_WEIGHT, 1 },
};

/* gemm_nt_q_gated: verbatim libref (ref_registry.cpp pGemmQGated/oGemmQGated) so
 * schema_same() passes. `act` is required here, unlike on the quantiser; `N` is the
 * weight's rows, twice the fused width, so the output is [M, N/2]. */
static const RadParamSpec pGemmNtQGated[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "N",     RAD_P_INT, RAD_REQUIRED },
    { "K",     RAD_P_INT, RAD_REQUIRED },
    { "group", RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
    { "act",   RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oGemmNtQGated[] = {
    { "a",         RAD_OPD_IN,     0 },
    { "a_scale",   RAD_OPD_IN,     0 },
    { "b",         RAD_OPD_WEIGHT, 0 },
    { "b_scale",   RAD_OPD_WEIGHT, 0 },
    { "q",         RAD_OPD_OUT,    0 },
    { "scale",     RAD_OPD_OUT,    0 },
    { "out_bf16",  RAD_OPD_OUT,    1 },
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
    /* v1.11: the e4m3 products a caller accepts. A permission, not a request -- libr3d's
     * rows compute the spelling they always computed; it is declared so the schema
     * matches libavx/libref, which refuse to load otherwise (ABI 26 rule). */
    { "prefill_fp8", RAD_P_INT, RAD_OPTIONAL },
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
/* The n-gram ban's own plane: one i32 {n-gram length, window} pair a row (n 0 = off,
 *  window <= 0 = the whole staged history). A plane beside `params` rather than fields
 *  in it, because the params row is the fixed 24-word RadSampleParams every sampler op
 *  shares. */
/* `whitelist` is optional: a bitmask plane a row (a SET bit exempts that column from
 *  the ban, the reference's whitelist), [M, ceil(n_vocab/32)] i32/u32, contiguous.
 *  Absent, nothing is exempt. */
static const RadOperandSpec oNgramBan[] = {
    { "logits",    RAD_OPD_INOUT, 0 },
    { "params",    RAD_OPD_IN,    0 },
    { "history",   RAD_OPD_IN,    0 },
    { "ngram",     RAD_OPD_IN,    0 },
    { "whitelist", RAD_OPD_IN,    1 },
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
/* rope_table: verbatim libref (ref_registry.cpp pRopeTable/oRopeTable). The 27B graph
 * issues it for its partial rotary (rot 64 of head_dim 256); libr4d declares the same
 * table, so a mixed rig dedupes it with first-declarer-wins. */
static const RadParamSpec pRopeTable[] = {
    { "M",             RAD_P_INT, RAD_REQUIRED },
    { "rot",           RAD_P_INT, RAD_REQUIRED },
    { "theta",         RAD_P_F64, RAD_REQUIRED },
    { "scale",         RAD_P_F64, RAD_REQUIRED },
    { "dtype",         RAD_P_STR, RAD_REQUIRED },
    { "yarn_factor",   RAD_P_F64, RAD_OPTIONAL },
    { "yarn_low",      RAD_P_F64, RAD_OPTIONAL },
    { "yarn_high",     RAD_P_F64, RAD_OPTIONAL },
    { "yarn_magnitude",RAD_P_F64, RAD_OPTIONAL },
    { "head_dim",      RAD_P_INT, RAD_OPTIONAL },
    { "ladder",        RAD_P_STR, RAD_OPTIONAL },
};
static const RadOperandSpec oRopeTable[] = {
    { "positions", RAD_OPD_IN,  0 },
    { "cos_sin",   RAD_OPD_OUT, 0 },
};
/* qk_norm_rope_gate (+ the kv_store fold): parameter and operand names, order and optionality
 * verbatim libr4d (r4d_rows.cpp pQkNormRopeGate/oQkNormRopeGate and the Kv pair) -- the graph
 * template binds them by name, so the spelling is the ABI. Not docs/OPS.md's `qk_norm_rope`:
 * different operands, and a precomputed cos/sin table instead of theta. The fp8 cache fold of
 * the donor has no row here (RDNA3 has no native E4M3 path); the bf16 fold is `k_out`'s and v's
 * two copies in the prologue's own blocks. */
static const RadParamSpec pQkNormRopeGate[] = {
    { "M",          RAD_P_INT, RAD_REQUIRED },
    { "head_dim",   RAD_P_INT, RAD_REQUIRED },
    { "n_head",     RAD_P_INT, RAD_REQUIRED },
    { "n_head_kv",  RAD_P_INT, RAD_REQUIRED },
    { "rot",        RAD_P_INT, RAD_REQUIRED },   /* rotary_dim, even and <= head_dim */
    { "q_dtype",    RAD_P_STR, RAD_REQUIRED },
    { "eps",        RAD_P_F64, RAD_OPTIONAL },
    { "wadd",       RAD_P_F64, RAD_OPTIONAL },   /* GemmaRMSNorm's `1 +`; 0 for a plain norm */
    { "cs_f32",     RAD_P_INT, RAD_OPTIONAL },   /* cos_sin table is f32 rather than bf16 */
    { "pos_i64",    RAD_P_INT, RAD_OPTIONAL },
};
/* The q half is optional (the k-only form): with `n_head` 0 there is no `q_gate`, `q_w` or
 * `q_out`, and every head is read from `k` at its row pitch. The gated attention template
 * passes no `gate_out`: `gate_quant_fp8` reads the gate in place in the projection. */
static const RadOperandSpec oQkNormRopeGate[] = {
    { "q_gate",    RAD_OPD_IN,     1 },
    { "k",         RAD_OPD_IN,     0 },
    { "cos_sin",   RAD_OPD_IN,     0 },
    { "positions", RAD_OPD_IN,     0 },
    { "q_w",       RAD_OPD_WEIGHT, 1 },
    { "k_w",       RAD_OPD_WEIGHT, 0 },
    { "q_out",     RAD_OPD_OUT,    1 },
    { "k_out",     RAD_OPD_OUT,    0 },
    { "gate_out",  RAD_OPD_OUT,    1 },
};
static const RadParamSpec pQkNormRopeGateKv[] = {
    { "M",          RAD_P_INT, RAD_REQUIRED },
    { "head_dim",   RAD_P_INT, RAD_REQUIRED },
    { "n_head",     RAD_P_INT, RAD_REQUIRED },
    { "n_head_kv",  RAD_P_INT, RAD_REQUIRED },
    { "rot",        RAD_P_INT, RAD_REQUIRED },
    { "q_dtype",    RAD_P_STR, RAD_REQUIRED },
    { "eps",        RAD_P_F64, RAD_OPTIONAL },
    { "wadd",       RAD_P_F64, RAD_OPTIONAL },
    { "cs_f32",     RAD_P_INT, RAD_OPTIONAL },
    { "pos_i64",    RAD_P_INT, RAD_OPTIONAL },
    { "block_size", RAD_P_INT, RAD_REQUIRED },
    { "kv_dtype",   RAD_P_STR, RAD_REQUIRED },
    { "k_scale",    RAD_P_F64, RAD_OPTIONAL },
    { "v_scale",    RAD_P_F64, RAD_OPTIONAL },
};
static const RadOperandSpec oQkNormRopeGateKv[] = {
    { "q_gate",       RAD_OPD_IN,     1 },
    { "k",            RAD_OPD_IN,     0 },
    { "cos_sin",      RAD_OPD_IN,     0 },
    { "positions",    RAD_OPD_IN,     0 },
    { "q_w",          RAD_OPD_WEIGHT, 1 },
    { "k_w",          RAD_OPD_WEIGHT, 0 },
    { "q_out",        RAD_OPD_OUT,    1 },
    { "k_out",        RAD_OPD_OUT,    0 },
    { "gate_out",     RAD_OPD_OUT,    1 },
    { "v",            RAD_OPD_IN,     0 },
    { "slot_mapping", RAD_OPD_IN,     0 },
    { "kv_cache",     RAD_OPD_INOUT,  0 },
};
/* gdn_gated_rmsnorm: verbatim libref (pGatedNorm/oGatedNorm). */
static const RadParamSpec pGatedNorm[] = {
    { "M",        RAD_P_INT, RAD_REQUIRED },
    { "channels", RAD_P_INT, RAD_REQUIRED },
    { "eps",      RAD_P_F64, RAD_REQUIRED },
    { "act",      RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oGatedNorm[] = {
    { "x", RAD_OPD_IN,     0 },
    { "z", RAD_OPD_IN,     0 },
    { "w", RAD_OPD_WEIGHT, 0 },
    { "o", RAD_OPD_OUT,    0 },
};
/* gdn_conv_update: verbatim libref (pConvUpdate/oConvUpdate). num_accepted is required by
 * the schema -- ref's comment: an op that lets the caller omit it is an op whose rollback
 * contract is optional. */
static const RadParamSpec pConvUpdate[] = {
    { "q_len",        RAD_P_INT, RAD_REQUIRED },
    { "head_k",       RAD_P_INT, RAD_REQUIRED },
    { "head_v",       RAD_P_INT, RAD_REQUIRED },
    { "conv_width",   RAD_P_INT, RAD_REQUIRED },
    { "max_query_len",RAD_P_INT, RAD_OPTIONAL },
};
static const RadOperandSpec oConvUpdate[] = {
    { "x",            RAD_OPD_IN,     0 },
    { "w",            RAD_OPD_WEIGHT, 0 },
    { "b",            RAD_OPD_WEIGHT, 1 },
    { "conv_state",   RAD_OPD_INOUT,  0 },
    { "state_idx",    RAD_OPD_IN,     0 },
    { "num_accepted", RAD_OPD_IN,     0 },
    { "cu",           RAD_OPD_IN,     0 },
    { "q",            RAD_OPD_OUT,    0 },
    { "k",            RAD_OPD_OUT,    0 },
    { "v",            RAD_OPD_OUT,    0 },
};
/* gdn_kkt_solve: verbatim libref (pGdnKktSolve/oGdnKktSolve). chunk is DERIVED with the
 * seq-chunk role, exactly libr4d's P_CHUNK spelling. */
static const RadParamSpec pKktSolve[] = {
    { "M",      RAD_P_INT, RAD_REQUIRED },
    { "head_k", RAD_P_INT, RAD_REQUIRED },
    { "chunk",  RAD_P_INT, RAD_DERIVED, RAD_PROLE_SEQ_CHUNK },
};
static const RadOperandSpec oKktSolve[] = {
    { "k",    RAD_OPD_IN,  0 },
    { "beta", RAD_OPD_IN,  0 },
    { "g",    RAD_OPD_IN,  0 },
    { "cu",   RAD_OPD_IN,  0 },
    { "A",    RAD_OPD_OUT, 0 },
};
/* gdn_chunk_scan: verbatim libref (pChunkScan/oChunkScan), including the optional h0/ht/
 * state_idx operands and the optional f64 `scale` and `state_fp16` parameters (ref reads the
 * first, ignores the second). */
static const RadParamSpec pChunkScan[] = {
    { "M",          RAD_P_INT,   RAD_REQUIRED },
    { "head_k",     RAD_P_INT,   RAD_REQUIRED },
    { "head_v",     RAD_P_INT,   RAD_REQUIRED },
    { "chunk",      RAD_P_INT,   RAD_DERIVED, RAD_PROLE_SEQ_CHUNK },
    { "scale",      RAD_P_F64,   RAD_OPTIONAL },
    { "state_fp16", RAD_P_INT,   RAD_OPTIONAL },
};
static const RadOperandSpec oChunkScan[] = {
    { "q",         RAD_OPD_IN,  0 },
    { "k",         RAD_OPD_IN,  0 },
    { "v",         RAD_OPD_IN,  0 },
    { "A",         RAD_OPD_IN,  0 },
    { "g",         RAD_OPD_IN,  0 },
    { "beta",      RAD_OPD_IN,  0 },
    { "h0",        RAD_OPD_IN,  1 },
    { "cu",        RAD_OPD_IN,  0 },
    { "o",         RAD_OPD_OUT, 0 },
    { "ht",        RAD_OPD_OUT, 1 },
    { "state_idx", RAD_OPD_IN,  1 },
};
/* gdn_recurrent_update: verbatim libref/libr4d (pGdnRecurrentUpdate/oGdnRecurrentUpdate).
 * q_len is required (the 27B pack records it, 1..128); the rest optional with the launcher
 * defaults matching ref's. norm_w is ROLE_WEIGHT like the oracle's, and its plain bf16 single
 * plane (this container's storage) needs no layout hook -- the row reads what it is given. */
static const RadParamSpec pRecurrent[] = {
    { "q_len",        RAD_P_INT, RAD_REQUIRED },
    { "head_k",       RAD_P_INT, RAD_REQUIRED },
    { "head_v",       RAD_P_INT, RAD_REQUIRED },
    { "n_head_v",     RAD_P_INT, RAD_OPTIONAL },
    { "scale",        RAD_P_F64, RAD_OPTIONAL },
    { "eps",          RAD_P_F64, RAD_OPTIONAL },
    { "l2_eps",       RAD_P_F64, RAD_OPTIONAL },
    { "softplus_thr", RAD_P_F64, RAD_OPTIONAL },
    { "act",          RAD_P_STR, RAD_OPTIONAL },
    { "state_fp16",   RAD_P_INT, RAD_OPTIONAL },
    { "state_form",   RAD_P_STR, RAD_OPTIONAL },
};
static const RadOperandSpec oRecurrent[] = {
    { "q",            RAD_OPD_IN,     0 },
    { "k",            RAD_OPD_IN,     0 },
    { "v",            RAD_OPD_IN,     0 },
    { "a",            RAD_OPD_IN,     0 },
    { "b",            RAD_OPD_IN,     0 },
    { "A_log",        RAD_OPD_WEIGHT, 0 },
    { "dt_bias",      RAD_OPD_WEIGHT, 0 },
    { "state",        RAD_OPD_INOUT,  0 },
    { "state_idx",    RAD_OPD_IN,     0 },
    { "num_accepted", RAD_OPD_IN,     0 },
    { "cu",           RAD_OPD_IN,     0 },
    { "z",            RAD_OPD_IN,     1 },
    { "norm_w",       RAD_OPD_WEIGHT, 1 },
    { "o",            RAD_OPD_OUT,    0 },
    { "o_q",          RAD_OPD_OUT,    1 },
    { "o_scale",      RAD_OPD_OUT,    1 },
};
/* gdn_conv_prep / gdn_conv_recurrent_update: the conv pair, verbatim libref (pConvPrep/
 * oConvPrep, pConvRecurrent/oConvRecurrent). chunk is DERIVED with the seq-chunk role as on
 * kkt_solve; z and norm_w are REQUIRED on the fused row to match the oracle's list -- the
 * launcher stays lenient, the schema does not. */
static const RadParamSpec pConvPrep[] = {
    { "M",            RAD_P_INT,  RAD_REQUIRED },
    { "head_k",       RAD_P_INT,  RAD_REQUIRED },
    { "head_v",       RAD_P_INT,  RAD_REQUIRED },
    { "chunk",        RAD_P_INT,  RAD_DERIVED, RAD_PROLE_SEQ_CHUNK },
    { "conv_width",   RAD_P_INT,  RAD_REQUIRED },
    { "l2_eps",       RAD_P_F64,  RAD_OPTIONAL },
    { "softplus_thr", RAD_P_F64,  RAD_OPTIONAL },
};
static const RadOperandSpec oConvPrep[] = {
    { "x",          RAD_OPD_IN,     0 },
    { "w",          RAD_OPD_WEIGHT, 0 },
    { "b",          RAD_OPD_WEIGHT, 1 },
    { "A_log",      RAD_OPD_WEIGHT, 0 },
    { "dt_bias",    RAD_OPD_WEIGHT, 0 },
    { "conv_state", RAD_OPD_INOUT,  0 },
    { "cu",         RAD_OPD_IN,     0 },
    { "a",          RAD_OPD_IN,     1 },
    { "b_gate",     RAD_OPD_IN,     1 },
    { "cache_idx",  RAD_OPD_IN,     1 },
    { "has_init",   RAD_OPD_IN,     1 },
    { "q",          RAD_OPD_OUT,    0 },
    { "k",          RAD_OPD_OUT,    0 },
    { "v",          RAD_OPD_OUT,    0 },
    { "g",          RAD_OPD_OUT,    0 },
    { "beta",       RAD_OPD_OUT,    0 },
};
static const RadParamSpec pConvFused[] = {
    { "q_len",        RAD_P_INT,  RAD_REQUIRED },
    { "head_k",       RAD_P_INT,  RAD_REQUIRED },
    { "head_v",       RAD_P_INT,  RAD_REQUIRED },
    { "conv_width",   RAD_P_INT,  RAD_REQUIRED },
    { "n_head_v",     RAD_P_INT,  RAD_OPTIONAL },
    { "scale",        RAD_P_F64,  RAD_OPTIONAL },
    { "eps",          RAD_P_F64,  RAD_OPTIONAL },
    { "l2_eps",       RAD_P_F64,  RAD_OPTIONAL },
    { "softplus_thr", RAD_P_F64,  RAD_OPTIONAL },
    { "act",          RAD_P_STR,  RAD_OPTIONAL },
    { "state_fp16",   RAD_P_INT,  RAD_OPTIONAL },
    { "state_form",   RAD_P_STR,  RAD_OPTIONAL },
};
static const RadOperandSpec oConvFused[] = {
    { "x",            RAD_OPD_IN,     0 },
    { "conv_w",       RAD_OPD_WEIGHT, 0 },
    { "conv_b",       RAD_OPD_WEIGHT, 1 },
    { "conv_state",   RAD_OPD_INOUT,  0 },
    { "conv_idx",     RAD_OPD_IN,     0 },
    { "a",            RAD_OPD_IN,     0 },
    { "b",            RAD_OPD_IN,     0 },
    { "A_log",        RAD_OPD_WEIGHT, 0 },
    { "dt_bias",      RAD_OPD_WEIGHT, 0 },
    { "state",        RAD_OPD_INOUT,  0 },
    { "state_idx",    RAD_OPD_IN,     0 },
    { "num_accepted", RAD_OPD_IN,     0 },
    { "cu",           RAD_OPD_IN,     0 },
    { "z",            RAD_OPD_IN,     0 },
    { "norm_w",       RAD_OPD_WEIGHT, 0 },
    { "o",            RAD_OPD_OUT,    0 },
    { "o_q",          RAD_OPD_OUT,    1 },
    { "o_scale",      RAD_OPD_OUT,    1 },
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
/* rmsnorm_quant_fp8 (verbatim libr4d pRmsNormQuantFp8 / oRmsNormQuantFp8). */
static const RadParamSpec pRmsNormQuantFp8[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    { "eps",   RAD_P_F64, RAD_REQUIRED },
    { "group", RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
    { "wadd",  RAD_P_F64, RAD_OPTIONAL },
};
static const RadOperandSpec oRmsNormQuantFp8[] = {
    { "x",        RAD_OPD_IN,     0 },
    { "residual", RAD_OPD_INOUT,  1 },   /* present = the fused_add_rms_norm form */
    { "w",        RAD_OPD_WEIGHT, 0 },
    { "q",        RAD_OPD_OUT,    0 },
    { "scale",    RAD_OPD_OUT,    0 },
    { "out_bf16", RAD_OPD_OUT,    1 },   /* layers with a bf16 consumer of the post-norm tensor */
};

/* Drafter ops (radiance 1.11 DFlash2 / MTP rerank). Schema tables verbatim libref's
 * pDflashConv/pDflashSelect/pLogitRerank + operands -- the loader refuses the whole
 * library on any disagreement (ABI 26 rule, learned the hard way on the retarget). */
static const RadParamSpec pDflashConv[] = {
    { "T",           RAD_P_INT, RAD_REQUIRED },
    { "hidden_size", RAD_P_INT, RAD_REQUIRED },
    { "taps",        RAD_P_INT, RAD_REQUIRED },
    { "group_size",  RAD_P_INT, RAD_REQUIRED },
    { "NG",          RAD_P_INT, RAD_REQUIRED },
    { "block_size",  RAD_P_INT, RAD_REQUIRED },
    { "dtype",       RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oDflashConv[] = {
    { "x",     RAD_OPD_IN,     0 },
    { "delta", RAD_OPD_IN,     0 },
    { "base",  RAD_OPD_WEIGHT, 0 },
    { "out",   RAD_OPD_OUT,    0 },
};
static const RadConstraint cDflashConv[] = {
    { "taps",        RAD_C_EQ,  2,   nullptr },
    { "group_size",  RAD_C_EQ,  16,  nullptr },
    { "hidden_size", RAD_C_DIV, 128, nullptr },
    { "dtype",       RAD_C_IN,  0,   "bf16" },
};
static const RadParamSpec pDflashSelect[] = {
    { "M",       RAD_P_INT, RAD_REQUIRED },
    { "steps",   RAD_P_INT, RAD_REQUIRED },
    { "top_k",   RAD_P_INT, RAD_REQUIRED },
    { "rank",    RAD_P_INT, RAD_REQUIRED },
    { "n_vocab", RAD_P_INT, RAD_REQUIRED },
    { "anchor_stride", RAD_P_INT, RAD_OPTIONAL },
    { "dtype",   RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oDflashSelect[] = {
    { "cand",   RAD_OPD_IN,     0 },
    { "unary",  RAD_OPD_IN,     0 },
    { "hp",     RAD_OPD_IN,     1 },
    { "anchor", RAD_OPD_IN,     0 },
    { "pred",   RAD_OPD_WEIGHT, 0 },
    { "succ",   RAD_OPD_WEIGHT, 0 },
    { "tokens", RAD_OPD_OUT,    0 },
};
static const RadConstraint cDflashSelect[] = {
    { "steps",   RAD_C_GE, 1,   nullptr },
    { "top_k",   RAD_C_GE, 1,   nullptr },
    { "top_k",   RAD_C_LE, 32,  nullptr },
    { "rank",    RAD_C_DIV, 32, nullptr },
    { "rank",    RAD_C_LE, 512, nullptr },
    { "dtype",   RAD_C_IN, 0,   "bf16" },
};
static const RadParamSpec pLogitRerank[] = {
    { "M",         RAD_P_INT, RAD_REQUIRED },
    { "R",         RAD_P_INT, RAD_REQUIRED },
    { "n_vocab",   RAD_P_INT, RAD_REQUIRED },
    { "n_embd",    RAD_P_INT, RAD_REQUIRED },
    { "vocab_off", RAD_P_INT, RAD_OPTIONAL },
    { "dtype",     RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec oLogitRerank[] = {
    { "x",      RAD_OPD_IN,     0 },
    { "head",   RAD_OPD_WEIGHT, 0 },
    { "idx_in", RAD_OPD_IN,     0 },
    { "idx",    RAD_OPD_OUT,    0 },
    { "val",    RAD_OPD_OUT,    0 },
    { "pairs",  RAD_OPD_OUT,    1 },
};
static const RadConstraint cLogitRerank[] = {
    { "M",       RAD_C_GE, 1,   nullptr },
    { "R",       RAD_C_GE, 1,   nullptr },
    { "R",       RAD_C_LE, 32,  nullptr },
    { "n_vocab", RAD_C_GE, 1,   nullptr },
    { "n_embd",  RAD_C_GE, 1,   nullptr },
    { "dtype",   RAD_C_IN, 0,   "bf16" },
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
    { "embed_lookup_q", pEmbedQ, (int)(sizeof(pEmbedQ)/sizeof(pEmbedQ[0])),
      oEmbedQ, (int)(sizeof(oEmbedQ)/sizeof(oEmbedQ[0])),
      "the same gather from a table quantised with one scale for all of it, or one per tile: "
      "x[m] = wte[id] * scale[b]. Qwen3.8-Flash-Next-FP8's 320-million-row n-gram embedding "
      "is stored with a single bf16 `weight_scale`, and it can be, because E4M3 carries its "
      "own exponent and a row's dynamic range is 2.3x; a block-fp8 embedding table instead "
      "carries a bf16 scale plane with one entry per 128x128 tile (the qwen35_fp8 token "
      "table gathers through this op). Padding and bounds behave exactly as embed_lookup's. "
      "`ahead` is the next prompt's ids, for a reader that fetches rows ahead of the step; "
      "a kernel that reads its table in place ignores it." },
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
    { "gemm_nt_q", pGemmNtQ, (int)(sizeof(pGemmNtQ)/sizeof(pGemmNtQ[0])),
      oGemmNtQ, (int)(sizeof(oGemmNtQ)/sizeof(oGemmNtQ[0])),
      "C[M,N] = A[M,K] @ dequant(Bq)[N,K]^T. `dtype` names the operand COMBINATION, so one "
      "op covers w4a16 through mxfp4a8 and an architecture plugin that changes quant changes "
      "one string." },
    { "gemm_nt_q_gated", pGemmNtQGated, (int)(sizeof(pGemmNtQGated)/sizeof(pGemmNtQGated[0])),
      oGemmNtQGated, (int)(sizeof(oGemmNtQGated)/sizeof(oGemmNtQGated[0])),
      "gemm_nt_q over a [2n, K] gate_up plane with gated_quant_fp8's pass folded into the "
      "epilogue: the outputs are the quantiser's E4M3 codes, its scale per (row, `group` "
      "columns) and optionally the bf16 product, and no [M, 2n] tensor is ever written. "
      "Byte-identical to the two ops it replaces, which is what makes it substitutable "
      "rather than a second numerical path." },
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
      { "sample_ngram_ban", pPenal, (int)(sizeof(pPenal)/sizeof(pPenal[0])),
        oNgramBan, (int)(sizeof(oNgramBan)/sizeof(oNgramBan[0])),
        "the windowed no-repeat-ngram ban: every token completing an n-gram the last "
        "`window` tokens of the row's history contain is set to -infinity. `ngram` is "
        "one i32 {n, window} pair a row; the window bounds the n-gram's START index, as "
        "in the reference, window <= 0 searches the whole span and n 0 is off. The "
        "optional `whitelist` bitmask exempts a token whose column bit is set; absent, "
        "nothing is exempt." },
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
      { "mul", pMnDtype, (int)(sizeof(pMnDtype)/sizeof(pMnDtype[0])),
        oBinary, (int)(sizeof(oBinary)/sizeof(oBinary[0])),
        "y = a * b, with the same one broadcast rule as `add`." },
      { "sigmoid", pMnDtype, (int)(sizeof(pMnDtype)/sizeof(pMnDtype[0])),
        oUnary, (int)(sizeof(oUnary)/sizeof(oUnary[0])),
        "y = 1 / (1 + exp(-x))." },
      { "rope_table", pRopeTable, (int)(sizeof(pRopeTable)/sizeof(pRopeTable[0])),
        oRopeTable, (int)(sizeof(oRopeTable)/sizeof(oRopeTable[0])),
        "the cos/sin plane a fused prologue reads instead of theta: cos then sin of "
        "`rope`'s ladder at each position's own row of a [max_ctx, rot] table; a negative "
        "position writes nothing." },
      { "qk_norm_rope_gate", pQkNormRopeGate, (int)(sizeof(pQkNormRopeGate)/sizeof(pQkNormRopeGate[0])),
        oQkNormRopeGate, (int)(sizeof(oQkNormRopeGate)/sizeof(oQkNormRopeGate[0])),
        "split [q|gate] per head, QK RMS norm over the head, partial NeoX RoPE and the gate "
        "copy, in one pass. Not docs/OPS.md's qk_norm_rope: different operands, and a "
        "precomputed cos/sin table instead of theta. `n_head` 0 is the K-ONLY FORM -- no "
        "`q_gate`, no `q_w`, no `q_out` -- for an input that is already one plain head per "
        "`head_dim` columns with nothing interleaved." },
      { "qk_norm_rope_gate_kv_store", pQkNormRopeGateKv, (int)(sizeof(pQkNormRopeGateKv)/sizeof(pQkNormRopeGateKv[0])),
        oQkNormRopeGateKv, (int)(sizeof(oQkNormRopeGateKv)/sizeof(oQkNormRopeGateKv[0])),
        "qk_norm_rope_gate followed by kv_store, in one launch: each k head's block also writes "
        "its rotated k and its v into the paged cache at the token's slot. The `k` kv_store "
        "reads is `k_out`, so the tail adds only `v`, `slot_mapping` and `kv_cache`." },
      { "gdn_gated_rmsnorm", pGatedNorm, (int)(sizeof(pGatedNorm)/sizeof(pGatedNorm[0])),
        oGatedNorm, (int)(sizeof(oGatedNorm)/sizeof(oGatedNorm[0])),
        "o = x * rsqrt(mean_channels(x^2) + eps) * w * act(z). `act` is \"silu\" or "
        "\"sigmoid\"." },
      { "gdn_conv_update", pConvUpdate, (int)(sizeof(pConvUpdate)/sizeof(pConvUpdate[0])),
        oConvUpdate, (int)(sizeof(oConvUpdate)/sizeof(oConvUpdate[0])),
        "The same convolution for a decode step against the rolling window: reads at "
        "num_accepted-1 and rewrites the buffer shifted down by one, so a speculative "
        "rejection is a change of read offset rather than a recompute. `num_accepted` is "
        "REQUIRED, because an op that lets the caller omit it is an op whose rollback "
        "contract is optional (spec \u00a710); a non-speculative decode passes ones." },
      { "gdn_kkt_solve", pKktSolve, (int)(sizeof(pKktSolve)/sizeof(pKktSolve[0])),
        oKktSolve, (int)(sizeof(oKktSolve)/sizeof(oKktSolve[0])),
        "A = (I + strict_lower(diag(beta) K K^T e^dg))^-1 per chunk." },
      { "gdn_recurrent_update", pRecurrent, (int)(sizeof(pRecurrent)/sizeof(pRecurrent[0])),
        oRecurrent, (int)(sizeof(oRecurrent)/sizeof(oRecurrent[0])),
        "The decode recurrence against the paged state cache, per-candidate form, gated RMS "
        "norm folded. Verbatim libref/libr4d so co-loaded schema_same() passes." },
      { "gdn_conv_prep", pConvPrep, (int)(sizeof(pConvPrep)/sizeof(pConvPrep[0])),
        oConvPrep, (int)(sizeof(oConvPrep)/sizeof(oConvPrep[0])),
        "The prefill preamble: depthwise conv with silu and its state cache, the q/k/v split, "
        "the l2 norm over each q and k head, the gate with its per-chunk cumsum, and beta. "
        "Verbatim libref so co-loaded schema_same() passes." },
      { "gdn_conv_recurrent_update", pConvFused, (int)(sizeof(pConvFused)/sizeof(pConvFused[0])),
        oConvFused, (int)(sizeof(oConvFused)/sizeof(oConvFused[0])),
        "gdn_conv_update followed by gdn_recurrent_update, with q, k and v never written; the "
        "rolling-window and state contracts are the two ops' own. Verbatim libref." },
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
      { "rmsnorm_quant_fp8", pRmsNormQuantFp8,
        (int)(sizeof(pRmsNormQuantFp8)/sizeof(pRmsNormQuantFp8[0])),
        oRmsNormQuantFp8, (int)(sizeof(oRmsNormQuantFp8)/sizeof(oRmsNormQuantFp8[0])),
        "RMS norm and the fp8 activation quantiser in one pass over the row. The bf16 norm "
        "output the unfused pair wrote is still written, and the codes are taken over it." },
    { "dflash_conv", pDflashConv, (int)(sizeof(pDflashConv)/sizeof(pDflashConv[0])),
      oDflashConv, (int)(sizeof(oDflashConv)/sizeof(oDflashConv[0])),
      "DFlash2 grouped dynamic depthwise convolution, coefficient add and both taps in "
      "one pass." },
    { "dflash_select", pDflashSelect,
      (int)(sizeof(pDflashSelect)/sizeof(pDflashSelect[0])),
      oDflashSelect, (int)(sizeof(oDflashSelect)/sizeof(oDflashSelect[0])),
      "DFlash2's candidate selector: the low-rank edge score between consecutive draft "
      "positions, and the greedy walk that turns a block of independent top-k sets into "
      "one path. Only the K edges leaving the previous pick are scored, because the walk "
      "is serial and the other K-1 rows of the score matrix are never read." },
    { "logit_rerank", pLogitRerank,
      (int)(sizeof(pLogitRerank)/sizeof(pLogitRerank[0])),
      oLogitRerank, (int)(sizeof(oLogitRerank)/sizeof(oLogitRerank[0])),
      "R candidate ids a row rescored EXACTLY against the full-precision head and "
      "re-sorted on those values -- the second half of a coarse top-R. `idx_in` "
      "carries global ids as row_topk reports them, `vocab_off` is this rank's first "
      "column, and an id outside its rows becomes a pad. Ties break on the LOWER id, "
      "the rule row_topk and row_topk_merge use, so a tie inside a shard and one "
      "between shards resolve alike." },
      };

/* The rmsnorm row. Field order is RadKernelInfo's positional form (rad_abi.h): after the prose,
 * constraints and (absent) tunables come init, fini, launch, scratch, layout, relayout, and
 * opd_shape -- null because a cold-calling tool takes operand shapes from libref's for the op.
 * `replaces` stays zero-initialised: this row is the unfused op, not a fusion. */

static const RadConstraint cBf16Row[] = {
    { "dtype", RAD_C_IN, 0, "bf16" },
};

static const RadConstraint cLogitsRow[] = {
    { "n_vocab", RAD_C_GE, 1,   nullptr },
    { "n_embd",  RAD_C_GE, 128, nullptr },
    { "dtype",   RAD_C_IN, 0, "bf16" },
};


/* logits_gemm bf16: verbatim libr4d cLogitsGemm = { C_DIV("n_embd", 8), C_IN("dtype", "bf16") }. */
static const RadConstraint cLogitsGemm[] = {
    { "n_embd", RAD_C_DIV, 8,    nullptr },
    { "dtype",  RAD_C_IN,  0,    "bf16" },
};

/* gemm_nt_q mxfp4a8: the format gate IS the dtype string (every operand dtypes follow
 * from it; the entry re-checks them). Other gemm_nt_q encodings are refused inside the
 * entry with RAD_E_UNSUPPORTED, so this constraint only keeps the selector honest. */
static const RadConstraint cMxfp4Row[] = {
    { "dtype", RAD_C_IN, 0, "mxfp4a8" },
};

/* gemm_nt_q fp8a8: same shape as the mxfp4 gate -- the dtype string IS the format, the
 * entry re-checks every operand dtype. The scale grid is 128-of-K on both sides (the
 * checkpoint's weight_block_size), and K a multiple of 128 so every WMMA group closes
 * on a scale block; ragged K resolves past this row to the reference. */
static const RadConstraint cFp8A8Row[] = {
    { "dtype", RAD_C_IN, 0, "fp8a8" },
    { "K",     RAD_C_DIV, 128, nullptr },
    { "group", RAD_C_EQ, 128, nullptr },
};

/* gemm_nt_q fp8a8 decode band (issue #4): the width row's band is M <= 8 -- decode and
 * speculative verify. Same format gates as the plain row (K closing on a scale block is
 * what lets the GEMV skip a group tail); the entry re-checks every dtype and the band's
 * own M bound. */
static const RadConstraint cFp8A8DecodeRow[] = {
    { "dtype", RAD_C_IN, 0, "fp8a8" },
    { "M",     RAD_C_LE, 8, nullptr },
    { "K",     RAD_C_DIV, 128, nullptr },
    { "group", RAD_C_EQ, 128, nullptr },
};

/* gemm_nt_q_mxfp4 decode band (issue #5): the width row's band is M <= 8 -- decode and
 * speculative verify. The 256-of-K spans need K a multiple of 128 (a tail span is then
 * exactly half-width); the entry re-checks every dtype, the band's own M bound, and the
 * uint2-per-lane byte alignment. */
static const RadConstraint cMxfp4DecodeRow[] = {
    { "dtype", RAD_C_IN, 0, "mxfp4a8" },
    { "M",     RAD_C_LE, 8, nullptr },
    { "K",     RAD_C_DIV, 128, nullptr },
};

/* gemm_nt_q w2a8 decode band: the DFlash2 draft head (the container declares the grid
 * only when the draft weight carries its 2-bit recipe). libr4d's cGemmNtW2A8 (K DIV 128,
 * N DIV 16, dtype w2a8) plus the drafter's own M ceiling and the checkpoint's group; the
 * M bound is 16, NOT the fp8/mxfp4 row's 8: the drafter's band collapses at 14 (verify
 * rows join the draft rows in the block pass), and a bucket with no device row sends the
 * WHOLE op host-side -- the head's joined scale plane then has no as-is spelling the
 * mapped tier may read, and startup dies on it. The MT=16 tile serves M <= 16.
 * The entry re-checks every dtype, the band's bound, and the dword walk. */
static const RadConstraint cW2A8DecodeRow[] = {
    { "dtype", RAD_C_IN,  0,    "w2a8" },
    { "K",     RAD_C_DIV, 128,  nullptr },
    /* PROBE: M constraint removed -- resolver-side M question. */
    /* NO N constraint: the drafter's gemm op carries no N parameter at issue -- N rides the
     * weight unit -- and a constraint on an absent key never holds (rad_types.h), which
     * drops the row from serve resolution entirely while kbench (fixture-fed N) happily
     * matches it (the 27B death of 2026-10-24: band to host, draft_head.scale onto the
     * mapped tier, rad.planes cannot map, exit). The entry re-checks N % 16 -- libr4d's
     * rule -- at plan time, where the real N is in hand. */
    { "group", RAD_C_EQ,  128,  nullptr },
};

/* gemm_nt_q_gated fp8a8: the composed row (tile into scratch, gated_quant epilogue)
 * sizes its arena at the band's M bound, so the band stops at 64 -- the whole dflash
 * drafter M range this container declares. act silu and sigmoid both compile in the
 * ported epilogue; K keeps the plain row's 128 rule. */
static const RadConstraint cFp8A8GatedRow[] = {
    { "dtype", RAD_C_IN, 0, "fp8a8" },
    { "M",     RAD_C_LE, 64, nullptr },
    { "K",     RAD_C_DIV, 128, nullptr },
    { "group", RAD_C_EQ, 128, nullptr },
    { "act",   RAD_C_IN, 0, "silu sigmoid" },
};

/* kv_store constraints, the positional form of libr4d's cKvStore = { C_GE("block_size", 1),
 * C_IN("kv_dtype", "bf16") }. */
static const RadConstraint cKvStore[] = {
    { "block_size", RAD_C_GE, 1, nullptr },
    { "kv_dtype",   RAD_C_IN, 0, "bf16" },
};

/* kv_store_fp8 (Q62): the E4M3 cache scatter. Same shape law as the bf16 row (the cache
 * strides are elements -- bytes for this dtype); the fold is ref_kv_store's 1/scale. */
static const RadConstraint cKvStoreFp8[] = {
    { "block_size", RAD_C_GE, 1, nullptr },
    { "kv_dtype",   RAD_C_IN, 0, "fp8_e4m3" },
};

/* qk_norm_rope_gate, the positional form of libr4d's cQkNormRopeGate = { C_IN("q_dtype",
 * "bf16"), C_LE("head_dim", 256), C_DIV("head_dim", 32) }. The Kv variants keep libr4d's
 * shape; the fp8 cache arm (Q62) joins as the kv pair's E4M3 sibling, the software-encoder
 * fold of r3d_qk_norm_rope_gate.hip. */
static const RadConstraint cQkNormRopeGate[] = {
    { "q_dtype",  RAD_C_IN, 0, "bf16" },
    { "head_dim", RAD_C_LE, 256, nullptr },
    { "head_dim", RAD_C_DIV, 32, nullptr },
};
static const RadConstraint cQkNormRopeGateKvBf16[] = {
    { "q_dtype",    RAD_C_IN, 0, "bf16" },
    { "head_dim",   RAD_C_LE, 256, nullptr },
    { "head_dim",   RAD_C_DIV, 32, nullptr },
    { "block_size", RAD_C_GE, 1, nullptr },
    { "kv_dtype",   RAD_C_IN, 0, "bf16" },
};
/* The fp8 pair of the fused row (Q62): identical shape law, E4M3 cache. */
static const RadConstraint cQkNormRopeGateKvFp8[] = {
    { "q_dtype",    RAD_C_IN, 0, "bf16" },
    { "head_dim",   RAD_C_LE, 256, nullptr },
    { "head_dim",   RAD_C_DIV, 32, nullptr },
    { "block_size", RAD_C_GE, 1, nullptr },
    { "kv_dtype",   RAD_C_IN, 0, "fp8_e4m3" },
};

/* libr3d is the only device library on RDNA3, so no in-plugin arbitration; the value only
 * needs to sit above a future fallback row. */
enum { R3D_PRIO_ONLY = 1000 };
enum { R3D_PRIO_ATTN_DECODE = 1100 };   /* split-KV rows win the decode band from the
                                           any-length tile; nothing else is ranked */
enum { R3D_PRIO_FP8_DECODE  = 1100 };   /* same law for gemm_nt_q: the width row wins
                                           M <= 8 from the m64 tile (issue #4) */
enum { R3D_PRIO_MXFP4_DECODE = 1100 };  /* the mxfp4 band's own width row (issue #5) */
enum { R3D_PRIO_MXFP4_Q8X    = 1200 };  /* the dot4 q8x arm wins the same band from the
                                           legacy width row by default (2026-10-25
                                           promotion); =0 withdraws it at registration */
enum { R3D_PRIO_W2A8_DECODE = 1100 };   /* same law: the width row owns the band; nothing
                                         * else in this plugin claims w2a8 at all */

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

/* Split-KV decode: the tuning axis is the split count -- 0 is the kernel's own
 * parallelism law (slices cut to fill the CUs), 8..64 measured overrides. 0 is
 * launchable by construction (the law's own floor is 1, and splits=1 runs the plain
 * tile through the same entry), which is all the loader requires of a default. */
static const int64_t kSplits[] = { 0, 8, 16, 32, 64 };
static const RadTunable tAttnSplit = { "splits", kSplits, 5, 0,
    "KV slices per (head, seq) computed in parallel; 0 = kernel law (fill the CUs)" };

/* The decode band these rows take from attn_paged_bf16: q_len <= 8 (decode and
 * speculative verify -- libr4d's decode-band bound, same spelling), window 0 (the row
 * serves a full KV walk; windowed decode stays with the tile), bf16 q, bf16/fp8-e4m3
 * cache. head_dim is one row per value, libr4d's convention: each is exactly a WMMA
 * template instantiation, and no IN set exists for integer keys. head_dim 256
 * (accept 2026-10-27) is bf16- and fp8-e4m3-cache since Q62 (the F8ST staging arms at
 * the fitted carve; R3D_HD256=0 still withdraws the whole family). */
static const RadConstraint cAttnDecodeH128[] = {
    RAD_CLE("q_len", 8),
    RAD_CEQ("window", 0),
    RAD_CEQ("head_dim", 128),
    RAD_CIN("q_dtype", "bf16"),
    RAD_CIN("kv_dtype", "bf16 fp8_e4m3"),
};
static const RadConstraint cAttnDecodeH64[] = {
    RAD_CLE("q_len", 8),
    RAD_CEQ("window", 0),
    RAD_CEQ("head_dim", 64),
    RAD_CIN("q_dtype", "bf16"),
    RAD_CIN("kv_dtype", "bf16 fp8_e4m3"),
};
static const RadConstraint cAttnDecodeH256[] = {
    RAD_CLE("q_len", 8),
    RAD_CEQ("window", 0),
    RAD_CEQ("head_dim", 256),
    RAD_CIN("q_dtype", "bf16"),
    RAD_CIN("kv_dtype", "bf16 fp8_e4m3"),   /* fp8@256 joins in Q62 (F8ST staging arms) */
};

/* rope: neox and gptj over [M] positions, plus imrope over [3, M] component-major positions
 * (the 27B trunk's attention) -- a subset of libr4d's five modes. A query for mrope/axial
 * resolves to no device row on this card, which is an honest failure at issue, not an
 * approximation. */
static const RadConstraint cRopeRow[] = {
    { "head_dim", RAD_C_GE, 1, nullptr },
    { "mode",     RAD_C_IN, 0, "neox gptj imrope" },
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
/* rope_table: the kernel's ladder and refusal cover every rot >= 2 even (odd refused at
 * launch, >1024 by the shared-array bound); yarn and the `head` ladder are refused there,
 * so nothing else is constrained here. */
static const RadConstraint cRopeTable[] = {
    { "rot", RAD_C_GE, 2, nullptr },
};
static const RadConstraint cGatedNorm[] = {
    { "channels", RAD_C_GE, 1, nullptr },
    { "act",      RAD_C_IN, 0, "silu sigmoid" },
};
static const RadConstraint cConvUpdate[] = {
    { "head_k",     RAD_C_GE, 1, nullptr },
    { "head_v",     RAD_C_GE, 1, nullptr },
    { "conv_width", RAD_C_GE, 1, nullptr },
};
/* kkt_solve: chunk 64 is the kernel's LDS bound (libr4d pins the same value; the derived
 * parameter resolves from A's shape). head_k stays open -- the dot chain walks any K. */
static const RadConstraint cKktSolve[] = {
    { "head_k", RAD_C_GE, 1, nullptr },
    RAD_CEQ("chunk", 64),
};
/* chunk_scan: head_k 128 is the register bound (the state column is a thread's register
 * array), head_v <= 128 the one-column-per-thread block; chunk 64 is the shared c64 value. */
static const RadConstraint cChunkScan[] = {
    RAD_CEQ("head_k", 128),
    { "head_v", RAD_C_GE, 1,    nullptr },
    { "head_v", RAD_C_LE, 128,  nullptr },
    RAD_CEQ("chunk", 64),
};
/* recurrent_update: the same register bounds as chunk_scan (r4d declares 128/128 outright;
 * the <= spelling keeps head_v under the one-column-per-thread block honest). The state_form
 * constraint admits both forms the arch block issues -- per_candidate (non-speculative decode)
 * and anchor (the speculative fold/replay against libr4d's factor-block layout) -- and the
 * rule that a constraint on a key the query lacks does not hold keeps unnamed callers with
 * the reference. */
static RadConstraint cRecurrent[] = {
    RAD_CEQ("head_k", 128),
    { "head_v", RAD_C_GE, 1,    nullptr },
    { "head_v", RAD_C_LE, 128,  nullptr },
    RAD_CIN("state_form", "per_candidate anchor"),
};
/* conv_prep: chunk 64 is the shared c64 value and the cumsum's reset unit; conv_width tops
 * at the kernel's LDS history bound; the l2 grid stages one head row, so head_k is bounded. */
static const RadConstraint cConvPrep[] = {
    RAD_CEQ("chunk", 64),
    { "head_k", RAD_C_GE, 1,    nullptr },
    { "head_k", RAD_C_LE, 4096, nullptr },
    { "conv_width", RAD_C_GE, 1, nullptr },
    { "conv_width", RAD_C_LE, 17, nullptr },
};
/* the fused pair: the recurrent row's register bounds, the window in registers (width <= 4),
 * and the same state_form scope gate -- both forms. */
static RadConstraint cConvFused[] = {
    RAD_CEQ("head_k", 128),
    { "head_v", RAD_C_GE, 1,    nullptr },
    { "head_v", RAD_C_LE, 128,  nullptr },
    { "conv_width", RAD_C_GE, 1, nullptr },
    { "conv_width", RAD_C_LE, 4, nullptr },
    RAD_CIN("state_form", "per_candidate anchor"),
};

extern "C" int r3d_rad_rmsnorm(const RadArgs*, RadStream);
extern "C" int r3d_rad_embed_lookup(const RadArgs*, RadStream);
extern "C" int r3d_rad_embed_lookup_q(const RadArgs*, RadStream);
extern "C" int r3d_rad_kv_store(const RadArgs*, RadStream);
extern "C" int r3d_rad_kv_store_fp8(const RadArgs*, RadStream);
extern "C" int r3d_rad_act_mul(const RadArgs*, RadStream);
extern "C" int r3d_rad_cast(const RadArgs*, RadStream);
extern "C" int r3d_rad_rope(const RadArgs*, RadStream);
extern "C" int r3d_rad_rope_table(const RadArgs*, RadStream);
extern "C" int r3d_rad_qk_norm_rope_gate(const RadArgs*, RadStream);
extern "C" int r3d_rad_qk_norm_rope_gate_kv_store(const RadArgs*, RadStream);
extern "C" int r3d_rad_qk_norm_rope_gate_kv_store_fp8(const RadArgs*, RadStream);
extern "C" int r3d_rad_gemm_nt(const RadArgs*, RadStream);
extern "C" int r3d_rad_logits_gemm(const RadArgs*, RadStream);
extern "C" int r3d_rad_gemm_nt_q(const RadArgs*, RadStream);
extern "C" int r3d_rad_gemm_nt_q_fp8(const RadArgs*, RadStream);
extern "C" int r3d_rad_gemm_nt_q_fp8_decode(const RadArgs*, RadStream);
extern "C" int r3d_rad_gemm_nt_q_mxfp4_decode(const RadArgs*, RadStream);
extern "C" int r3d_rad_gemm_nt_q_mxfp4_decode_q8x(const RadArgs*, RadStream);
extern "C" bool r3d_mxfp4_q8x_off(void);   /* r3d_gemm_mxfp4.hip: default ON, literal
   "0" withdraws. extern "C" BOTH sides: this file's body sits inside the header's
   extern "C" {" block (line 14), so a bare declaration silently asks for C linkage
   while a C++ definition mangles -- undefined symbol only at dlopen, never at link. */
extern "C" int r3d_rad_gemm_nt_q_gated_fp8(const RadArgs*, RadStream);
extern "C" int64_t r3d_rad_gemm_nt_q_gated_fp8_scratch(const RadArgs*);
extern "C" int r3d_rad_logits_gemm_e4m3(const RadArgs*, RadStream);
extern "C" int r3d_layout_head(const RadParam*, int, int, const RadEncoding*, const int*,
                               const RadTensor*, int, RadLayout*);
extern "C" int r3d_relayout_head(const RadParam*, int, int, const RadEncoding*, const int*,
                                 const RadTensor*, int, void*, int64_t);
extern "C" int r3d_unrelayout_head(const RadParam*, int, int, const RadEncoding*, const int*,
                                   const void*, int64_t, const RadTensor*, int);
extern "C" int r3d_rad_attn_paged(const RadArgs*, RadStream);
extern "C" int r3d_rad_attn_paged_decode(const RadArgs*, RadStream);
extern "C" int64_t r3d_rad_attn_decode_scratch(const RadArgs*);
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
extern "C" int r3d_rad_sample_ngram_ban(const RadArgs*, RadStream);
extern "C" int r3d_rad_logit_rerank_e4m3(const RadArgs*, RadStream);
extern "C" int r3d_rad_gemm_nt_q_w2a8_decode(const RadArgs*, RadStream);
extern "C" int r3d_layout_gemm_w2a8(const RadParam*, int, int, const RadEncoding*,
                                    const int*, const RadTensor*, int, RadLayout*);
extern "C" int r3d_relayout_gemm_w2a8(const RadParam*, int, int, const RadEncoding*,
                                      const int*, const RadTensor*, int, void*, int64_t);
extern "C" int r3d_unrelayout_gemm_w2a8(const RadParam*, int, int, const RadEncoding*,
                                        const int*, const void*, int64_t,
                                        const RadTensor*, int);
extern "C" int r3d_rad_dflash_conv(const RadArgs*, RadStream);
extern "C" int r3d_rad_dflash_select(const RadArgs*, RadStream);
extern "C" int r3d_rad_logit_rerank(const RadArgs*, RadStream);
extern "C" int r3d_rad_sample_xtc(const RadArgs*, RadStream);
extern "C" int r3d_rad_sample_logprobs(const RadArgs*, RadStream);
extern "C" int r3d_rad_quant_act_fp8(const RadArgs*, RadStream);
extern "C" int r3d_rad_gated_quant_fp8(const RadArgs*, RadStream);
extern "C" int r3d_rad_add(const RadArgs*, RadStream);
extern "C" int r3d_rad_mul(const RadArgs*, RadStream);
extern "C" int r3d_rad_sigmoid(const RadArgs*, RadStream);
extern "C" int r3d_rad_gather_rows(const RadArgs*, RadStream);
extern "C" int r3d_rad_gdn_gated_rmsnorm(const RadArgs*, RadStream);
extern "C" int r3d_rad_gdn_conv_update(const RadArgs*, RadStream);
extern "C" int r3d_rad_gdn_kkt_solve(const RadArgs*, RadStream);
extern "C" int r3d_rad_gdn_chunk_scan(const RadArgs*, RadStream);
extern "C" int r3d_rad_gdn_recurrent_update(const RadArgs*, RadStream);
extern "C" int r3d_rad_gdn_conv_prep(const RadArgs*, RadStream);
extern "C" int r3d_rad_gdn_conv_recurrent_update(const RadArgs*, RadStream);
extern "C" int r3d_rad_silu_mul(const RadArgs*, RadStream);
extern "C" int r3d_rad_quant_act_i8(const RadArgs*, RadStream);
extern "C" int r3d_rad_row_topk(const RadArgs*, RadStream);
extern "C" int r3d_rad_rmsnorm_quant_fp8(const RadArgs*, RadStream);

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
static const RadConstraint cRmsNormQuantFp8[] = {
    { "n",     RAD_C_DIV, 4,    nullptr },
    { "group", RAD_C_EQ,  128,  nullptr },   /* R4D_FP8_BLOCK */
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

{ "embed_lookup_q_bf16", "embed_lookup_q", "vocab",
  "quantised embedding gather, four entries a thread (Q61b)",
  "any n_embd and n_vocab; a negative or out-of-range id writes a zero row",
  "i32/i64 tokens, an E4M3 table with a bf16 or f32 scale -- one entry for the whole table "
  "or a per-tile plane (the block-fp8 embedding of qwen35_fp8) -- or a plain bf16 table, "
  "bf16 output; the product is one f32 rounding (exact there) and one RTNE narrowing, "
  "bit-equal to libref's embed_gather",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cBf16Row, 1, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_embed_lookup_q },

{ "kv_store_bf16", "kv_store", "attn",
  "scatter k and v into the paged cache, one workgroup a token",
  "cache exactly [n_blocks, kv_heads, block_size, 2*head_dim]; a negative or out-of-range slot "
  "stores nothing",
  "bf16 k / v / cache, i32 slot mapping",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cKvStore, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_kv_store },

{ "kv_store_fp8", "kv_store", "attn",
  "scatter k and v into the E4M3 paged cache, one workgroup a token (Q62)",
  "cache exactly [n_blocks, kv_heads, block_size, 2*head_dim] fp8_e4m3; a negative or "
  "out-of-range slot stores nothing",
  "bf16 k / v, fp8_e4m3 cache, i32 slot mapping; stored value = ref_kv_store's "
  "value * (1/k_scale) through the ABI's saturating RTNE encoder",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cKvStoreFp8, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_kv_store_fp8 },

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

{ "rope_table_f32", "rope_table", "attn",
  "cos/sin of the rope ladder per position row, one block per position, the ladder "
  "built once in shared per row",
  "rot even in [2, 1024]; [max_ctx, rot] table with unit last stride; a position "
  "outside the table writes its row nothing (ref skips it the same); yarn and the "
  "`head` ladder have no row here",
  "i32/i64 positions, f32 or bf16 table, fp32 angles through the accurate sincosf",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cRopeTable, 1, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_rope_table },

{ "qk_norm_rope_gate", "qk_norm_rope_gate", "attn",
  "the gated-attention QKV preamble in one pass: split [q|gate] per head, QK RMS norm over "
  "the head, partial NeoX RoPE from the shared cos/sin table and the gate copy; byte-identical "
  "to the rmsnorm+rope ops it replaces (r3d_qk_norm_rope_gate.hip header for the anchors)",
  "head_dim 64, 128 or 256; rotary_dim even and <= head_dim; NeoX style; n_head may be 0, "
  "which is the K-only form and serves an input with no gate interleaved",
  "bf16 in and out; the norm output is rounded to bf16 before the rotation, as the unfused "
  "reference does by storing it; f32 or bf16 cos/sin and gains, i32 or i64 positions",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cQkNormRopeGate, 3, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_qk_norm_rope_gate },

{ "qk_norm_rope_gate_kv_store_bf16", "qk_norm_rope_gate_kv_store", "attn",
  "qk_norm_rope_gate, and kv_store into a bf16 paged cache from the same launch -- the two "
  "bytes per k-head element that kv_store_bf16 would copy, written from the block that "
  "already holds them",
  "qk_norm_rope_gate's shapes; cache exactly [n_blocks, kv_heads, block_size, 2*head_dim] "
  "and one slot a row; a negative or out-of-range slot stores nothing",
  "bf16 in and out, bf16 cache, i32 slot mapping",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cQkNormRopeGateKvBf16, 5, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_qk_norm_rope_gate_kv_store },

{ "qk_norm_rope_gate_kv_store_fp8", "qk_norm_rope_gate_kv_store", "attn",
  "qk_norm_rope_gate, and kv_store into an E4M3 paged cache from the same launch (Q62) -- "
  "the two bytes per k-head element that kv_store_fp8 would write, encoded from the block "
  "that already holds them",
  "qk_norm_rope_gate's shapes; cache exactly [n_blocks, kv_heads, block_size, 2*head_dim] "
  "and one slot a row; a negative or out-of-range slot stores nothing",
  "bf16 in and out, fp8_e4m3 cache, i32 slot mapping; the stored byte is the bf16 value "
  "kv_store_fp8 would read, re-widened and scaled exactly as ref_kv_store does it",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cQkNormRopeGateKvFp8, 5, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_qk_norm_rope_gate_kv_store_fp8 },

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

/* logits_gemm: the e4m3 head with one float scale per 128 columns of K (the
 * layout hook admits exactly that joined form; the launcher pins the scale
 * grid's row block to 1 or 128). Ragged K without a four-byte row walk, and
 * int8 heads, are refused -- the host libraries keep those. Declared AFTER the
 * bf16 head row on purpose: a plain head must be claimed by the bf16 row (an
 * unsupported hook answer is "store as is", not a decline -- declaring first
 * here would win the plain band and refuse it at launch, dropping serve to
 * host). An encoded band declines the hookless bf16 row at declare time
 * anyway, so this row is the only one left standing for it. */
{ "logits_gemm_e4m3", "logits_gemm", "vocab",
  "bandwidth GEMV over the joined lm_head: warp per head row, four codes per "
  "lane per step, x staged per 128-column block through LDS; the decoded e4m3 "
  "times the block scale is one fma chain per output in f32, and the wavefront "
  "folds its 32 partials through a butterfly",
  "codes [n_vocab, n_embd] e4m3 dense, scale per (1 or 128 rows, 128 cols); M "
  "derives from numel(y)/n_vocab, groups of up to 8 rows ride grid.y",
  "bf16 x, joined e4m3+bf16/f16/f32 head, f32 logits",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cLogitsRow, 3, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_logits_gemm_e4m3,
  nullptr, r3d_layout_head, r3d_relayout_head, nullptr, r3d_unrelayout_head },

{ "gemm_nt_q_mxfp4", "gemm_nt_q", "gemm",
  "mxfp4a8 WMMA tile: e2m1 nibbles decoded and E8M0-folded into bf16 B fragments per "
  "32-K group, fp8_e4m3 activations decoded exactly, fp32 accumulators flushed and "
  "row-scaled by a_scale per 128-K group through LDS",
  "b:32 / a:128 scale grids only; nibble rows byte-packed; ragged tiles load as zero "
  "and are never stored; M derives from numel(y)/N",
  "fp8_e4m3 a with f32 a_scale, e2m1 b with E8M0 b_scale, bf16 y; b_ref accepted and "
  "ignored (both host libraries do); a_sum refused",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cMxfp4Row, 1, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_gemm_nt_q },

{ "gemm_nt_q_fp8a8", "gemm_nt_q", "gemm",
  "fp8a8 WMMA tile: e4m3 codes on both operands decoded through an exact bf16 LUT, one "
  "128-of-K scale group per WMMA burst, the group partial flushed through LDS as "
  "(sub * a_scale) * b_scale in the oracle's order, bf16 store",
  "K a multiple of 128 (the scale grid must close on a whole block); N any (ragged "
  "columns load zero and are never stored); M derives from numel(y)/N; b_scale either "
  "the 128x128 checkpoint block grid or a per-row grid",
  "fp8_e4m3 a with f32 a_scale per (row, 128 K), fp8_e4m3 b with bf16 scale per 128x128 "
  "tile (or per row), bf16 y; a_sum and b_ref refused (as ref does at this encoding)",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cFp8A8Row, 3, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_gemm_nt_q_fp8 },

{ "gemm_nt_q_fp8a8_decode", "gemm_nt_q", "gemm",
  "fp8a8 decode-band width row: warp per weight row, four codes per lane (one aligned "
  "dword), the activation rows staged per 128-of-K group through LDS, every weight byte "
  "read exactly once; the group partial folds through a butterfly and flushes as "
  "(sub * a_scale) * b_scale in the oracle's order",
  "M <= 8 (decode and speculative verify -- the band the m64 tile wastes at ragged M); "
  "K a multiple of 128; N any (dead warps clamp their load row and never store); "
  "b_scale either the 128x128 checkpoint block grid or a per-row grid",
  "fp8_e4m3 a with f32 a_scale per (row, 128 K), fp8_e4m3 b with bf16 scale per 128x128 "
  "tile (or per row), bf16 y; a_sum and b_ref refused; a B whose row pitch or base "
  "breaks the dword walk resolves to the m64 row",
  RAD_DOMAIN_DEVICE, R3D_PRIO_FP8_DECODE,
  cFp8A8DecodeRow, 4, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_gemm_nt_q_fp8_decode },

{ "gemm_nt_q_mxfp4_decode", "gemm_nt_q", "gemm",
  "mxfp4a8 decode-band width row: warp per weight row, eight codes per lane (one "
  "aligned uint2), the activation rows decoded per 256-of-K span through LDS, every "
  "weight byte read exactly once; each 4-lane quad owns one 32-of-K scale group and "
  "flushes (sub * a_scale) * 2^(e8m0-127) in the oracle's order -- the per-32 group "
  "structure the m64 tile reassociates away",
  "M <= 8 (decode and speculative verify -- the band the m64 tile wastes at ragged "
  "M); K a multiple of 128 (the 256-of-K spans close, a tail span is half-width); "
  "N any (dead warps clamp their load row and never store); M derives from numel(y)/N",
  "fp8_e4m3 a with f32 a_scale per (row, 128 K), e2m1 b with E8M0 b_scale per "
  "(row, 32 K), bf16 y; b_ref accepted and ignored (both host libraries do); a_sum "
  "refused; a B whose base breaks the uint2 walk resolves to the m64 tile",
  RAD_DOMAIN_DEVICE, R3D_PRIO_MXFP4_DECODE,
  cMxfp4DecodeRow, 3, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_gemm_nt_q_mxfp4_decode },

{ "gemm_nt_q_mxfp4_decode_q8x", "gemm_nt_q", "gemm",
  "mxfp4a8 decode-band dot4 q8x arm (DEFAULT ON, 2026-10-25 promotion): one wave32 per "
  "weight row over the plugin's own interleaved byte plane (one i8 = e2m1 code x2 with "
  "the e8m0 group scales woven as 136B headers, repacked lazily once per tensor on the "
  "plugin stream), i8 activations requantized per dispatch, four signed v_dot4_i32_iu8 "
  "dots per lane per sixteen codes; the per-group flush keeps the shipped quant_dot "
  "order. Measured +23.5% serve decode median over the legacy walk (2026-10-24 pair)",
  "same band as the legacy width row (M <= 8, K % 128) plus the arm's own plan gate "
  "K % 256 == 0 -- K % 256 == 128 and every arm-unavailable geometry (plane build, "
  "VRAM headroom, launch) serve the legacy walk IN-PLACE from the same entry; the "
  "plane cache is per (B, BS, N, K), never freed",
  "fp8_e4m3 a with f32 a_scale per (row, 128 K), e2m1 b with E8M0 b_scale per "
  "(row, 32 K), bf16 y; b_ref accepted and ignored; a_sum refused. R3D_MXFP4_Q8X=0 "
  "(literal \"0\") withdraws this row at the registration gate and the band resolves "
  "to the legacy decode row -- the kill switch is a re-resolve, never an entry-time "
  "refusal (issue #4 law); R3D_NO_FP4_DEC / R3D_NO_MXFP4 withdraw this row too",
  RAD_DOMAIN_DEVICE, R3D_PRIO_MXFP4_Q8X,
  cMxfp4DecodeRow, 3, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_gemm_nt_q_mxfp4_decode_q8x },

{ "gemm_nt_q_w2a8_decode", "gemm_nt_q", "gemm",
  "w2a8 decode-band GEMV (the DFlash2 draft head): warp per weight row. The q8 arm "
  "walks the codes repacked to one byte a code (the plugin's own cached plane, built "
  "once per head) and folds sixteen codes per lane with four v_dot4_i32_iu8 dots; "
  "R3D_NO_W2A8_Q8X=1 falls back to the u2 walk -- one aligned dword per lane per "
  "four scale groups, sixteen 2-bit codes per lane, fmaf chains. Either arm stages "
  "the int8 activation per 512-of-K window through LDS with quant_act_i8's step "
  "order de-permuted in the staging index; each group's partial folds through an "
  "octet butterfly and flushes as (sub * a_scale) * b_scale in the oracle's order, "
  "sub = is - zero * a_sum first -- every value before the scales is an exact "
  "integer, so both arms are bit-exact to ref_gemm's gemm_zp, not merely close",
  "M <= 16 (the drafter's whole gemm_nt_q band -- its block pass collapses M at 14, "
  "verify rows included; a bucket above this would strand the op host-side, which its "
  "joined scale plane cannot survive; K a multiple of 128; N a multiple of 16 (libr4d's "
  "rule); group exactly 128; M derives from numel(y)/N; a B whose base or row pitch "
  "breaks the dword walk resolves past the row, as everything past the band does -- "
  "to the host",
  "int8 a with an f32 per-row scale (quant_act_i8's step order), u2 b (the codes "
  "plane stored as it is) with an f16 scale and a u8/u4 zero per 128 K in the "
  "canonical rad.planes join (libref's own layout), f32 a_sum = quant_act_i8's "
  "group-major [K/128][Mpad] sums (REQUIRED -- the zero point's correction), bf16 "
  "y; b_ref refused; pb1/pb2 refused; an encoding transform declines at declare",
  RAD_DOMAIN_DEVICE, R3D_PRIO_W2A8_DECODE,
  cW2A8DecodeRow, 3, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_gemm_nt_q_w2a8_decode,
  nullptr, r3d_layout_gemm_w2a8, r3d_relayout_gemm_w2a8, nullptr,
  r3d_unrelayout_gemm_w2a8 },

{ "gemm_nt_q_gated_fp8a8", "gemm_nt_q_gated", "gemm",
  "the fp8a8 tile writing its bf16 gate_up plane into the launch scratch, closed by the "
  "ported gated_quant_fp8 epilogue in the same launch -- the byte-identical composition "
  "of the two shipped rows (libr4d's fuse-check precedent); no [M,2n] tensor reaches the "
  "caller",
  "M <= 64 (the scratch is sized at the band's bound; dflash's drafter band); K a "
  "multiple of 128; N = 2n with n a whole number of 128-column scale groups in practice "
  "(the epilogue's amax closes per group inside the row); M derives from numel(q)/n",
  "fp8_e4m3 a with f32 a_scale per (row, 128 K), fp8_e4m3 b with bf16 scale per 128x128 "
  "tile, e4m3 q + f32 scale out, optional bf16 product; act silu or sigmoid; a_sum refused",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cFp8A8GatedRow, 5, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_gemm_nt_q_gated_fp8,
  r3d_rad_gemm_nt_q_gated_fp8_scratch },

{ "attn_paged_bf16", "attn_paged", "attn",
  "paged causal / sliding-window attention, one workgroup per (query row, head), any "
  "query length, varlen, split-free (no scratch)",
  "any head_dim to 1024, any gqa, any block size, window and causal masking, ragged "
  "batches via cu_seqlens, unmapped (negative) block table entries skipped",
  "bf16 query, bf16 / f32 / fp8-e4m3 kv cache with optional per-(seq,head) descales, "
  "bf16 or f32 out, f32 softmax with the max subtracted",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cAttnPaged, 3, nullptr, 0, nullptr,
  /* The decode scratch hook rides on THIS row because it is the row the engine
   * actually selects at boot (see the live-band comment in r3d_attn_paged.hip):
   * the split-K entry is reached from here at issue time, so the arena it needs
   * must be sized by the row that was resolved. The hook is the split-K law's
   * own closed form and answers 0 for shapes it cannot serve. */
  nullptr, nullptr, r3d_rad_attn_paged, r3d_rad_attn_decode_scratch },

{ "attn_paged_decode_bf16kv_h128", "attn_paged", "attn",
  "split-KV decode attention: the WMMA tile's two passes over each of `splits` KV slices "
  "in parallel, un-normalized partials (O, m, l) in the engine arena, one merge kernel "
  "rescaling them per output row; splits=1 IS the plain tile, same body, same tail math",
  "decode and speculative verify (q_len <= 8), window 0, head_dim 128, any gqa, any "
  "block size, ragged batches via cu_seqlens, unmapped block table entries skipped",
  "bf16 query, bf16 / fp8-e4m3 kv cache with optional per-(seq,head) descales, bf16 or "
  "f32 out, f32 partials and merge",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ATTN_DECODE,
  cAttnDecodeH128, 5, &tAttnSplit, 1, nullptr,
  nullptr, nullptr, r3d_rad_attn_paged_decode, r3d_rad_attn_decode_scratch },

{ "attn_paged_decode_bf16kv_h64", "attn_paged", "attn",
  "split-KV decode attention, head_dim 64: as attn_paged_decode_bf16kv_h128",
  "decode and speculative verify (q_len <= 8), window 0, head_dim 64, any gqa, any "
  "block size, ragged batches via cu_seqlens, unmapped block table entries skipped",
  "bf16 query, bf16 / fp8-e4m3 kv cache with optional per-(seq,head) descales, bf16 or "
  "f32 out, f32 partials and merge",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ATTN_DECODE,
  cAttnDecodeH64, 5, &tAttnSplit, 1, nullptr,
  nullptr, nullptr, r3d_rad_attn_paged_decode, r3d_rad_attn_decode_scratch },

{ "attn_paged_decode_bf16kv_h256", "attn_paged", "attn",
  "split-KV decode attention, head_dim 256: as attn_paged_decode_bf16kv_h128 on the "
  "m2 fitted body at SPLITK=1 (issue #11 m3); bf16 and fp8_e4m3 caches (Q62 staging arms)",
  "decode and speculative verify (q_len <= 8), window 0, head_dim 256, any gqa, any "
  "block size, ragged batches via cu_seqlens, unmapped block table entries skipped",
  "bf16 query, bf16 kv cache with optional per-(seq,head) descales, bf16 or f32 out, "
  "f32 partials and merge",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ATTN_DECODE,
  cAttnDecodeH256, 5, &tAttnSplit, 1, nullptr,
  nullptr, nullptr, r3d_rad_attn_paged_decode, r3d_rad_attn_decode_scratch },

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

{ "mul_act", "mul", "band",
  "elementwise product over dense [rows, n] rows, one thread a lane",
  "f32/bf16 operands; b broadcasts along rows only at exactly n elements; holey row "
  "pitches are refused (the kernel walks the range linearly, ref's offlin not)",
  "f32/bf16 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cBandMn, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_mul },

{ "sigmoid_act", "sigmoid", "band",
  "the double-rounded sigmoid band, walked linearly over numel(Y)",
  "f32/bf16; dense layouts only (see r3d_band.hip on why a pitch wider than n is "
  "refused here and not in `add`)",
  "f32/bf16 in and out, exp through the same double-rounded form the quantizers use",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cBandMn, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_sigmoid },

{ "gather_rows_act", "gather_rows", "band",
  "row gather with the zero-row padding convention",
  "f32/bf16 rows, i32 indices; a negative index writes a zero row; an index >= the "
  "source rows ALSO writes zeros rather than failing the launch (a device kernel cannot "
  "pre-validate like the host reference; documented deviation)",
  "f32/bf16 x and y, i32 idx",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cBandMn, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_gather_rows },

/* ---- the Gated DeltaNet family (see r3d_gdn.hip): 48 of the qwen3.5-27B trunk's 64
 * layers run on these. Op schemas are verbatim libref, so libr4d (gfx12) and libr3d
 * dedupe them cleanly on a mixed rig -- and once the last GDN row lands, a 27B program
 * splits per-op between the two libraries instead of refusing to load. */

{ "gdn_gated_rmsnorm_bf16", "gdn_gated_rmsnorm", "gdn",
  "channel-RMS norm gated by silu/sigmoid of a companion plane, one block per "
  "(token, value head) row",
  "channels up to 65535; x and o dense-row [rows, channels] (a rank-2 pitch wider than "
  "the row is fine, wider unit-strides are not); z may be a column slice of a fused "
  "[rows, heads*channels] row -- the engine's own gate_rowoff walk",
  "f32/bf16 x/z/o, f32-or-bf16 weight read as stored, f32 row sum and scale chain",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cGatedNorm, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_gdn_gated_rmsnorm },

{ "gdn_conv_update_bf16", "gdn_conv_update", "gdn",
  "decode-step depthwise conv with silu over the rolling window, one block per "
  "(sequence, channel); the window reads at num_accepted-1 and rewrites shifted down "
  "by one, so rejection is a read-offset change, not a recompute",
  "conv_width up to 17 taps; x dense-row [T, conv_dim] (wider pitch fine, wider "
  "unit-stride not); conv_state [slots, conv_dim, state_len] or flat-tight; index "
  "tensors (state_idx, num_accepted, cu) must be fully dense; a window check that "
  "fails skips its sequence instead of failing the op",
  "f32/bf16 x/w/b/conv_state/q/k/v; i32/u32/i64 index reads; the width-tap f32 chain "
  "runs in ref's ascending order over exact loads -- bit-exact by construction",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cConvUpdate, 3, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_gdn_conv_update },

{ "gdn_kkt_solve_bf16", "gdn_kkt_solve", "gdn",
  "chunk preamble A = (I + strict_lower(diag(beta) K K^T e^dg))^-1, one block per "
  "(chunk, sequence, v head); the inverse is ref's forward substitution with threads "
  "owning columns, so every f32 chain runs in ref's order",
  "chunk 64 (LDS bound; the derived parameter resolves from A's shape, other chunks are "
  "refused); any head_k; k/A dense last stride with a wider row pitch fine; beta/g dense "
  "within the row; cu fully dense; blocks past a sequence's end exit empty",
  "f32/bf16 k and A out, f32-or-bf16 beta/g read as stored; the gram dot is a d-ascending "
  "f32 chain over exact widening and the substitution an l-ascending one -- bit-exact by "
  "construction, the scalar sibling of libr4d's WMMA blocked merge",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cKktSolve, 2, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_gdn_kkt_solve },

{ "gdn_chunk_scan_bf16", "gdn_chunk_scan", "gdn",
  "the chunked delta-rule scan W = beta(V - e^g K S^T), V' = A W, O = scale e^g Q S^T + sA V', "
  "S = e^{g_last} S + V'^T K', one block per (sequence, v head); the chunk recurrence is "
  "serial by nature, so the transposed f32 state lives in one-column-per-thread register "
  "arrays and every chain walks the oracle's order",
  "head_k 128 (register bound: the state column IS a thread's register array), head_v up to "
  "128 (one value column per thread), chunk 64 (the pack's and the c64 family's; derived from "
  "A's shape when the parameter is absent); q/k/v/A/o dense last stride with wider row pitch "
  "fine; h0/ht strides read off the tensor (a paged state cache is page-padded); state_idx "
  "rows may be wider than one slot (column 0 is the slot), a negative slot skips the row's "
  "outputs and state exactly as the reference's continue does",
  "f32/bf16 q/k/v/A/o, f32-or-bf16 g/beta read as stored, f32 h0/ht state planes; every "
  "ascending-d dot, the descending-in-place V' = A W, and the state update are f32 chains in "
  "the oracle's order over exact widening -- bit-exact by construction, the scalar sibling of "
  "libr4d's k128_c64 row",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cChunkScan, 4, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_gdn_chunk_scan },

{ "gdn_recurrent_update_bf16", "gdn_recurrent_update", "gdn",
  "the decode delta rule S = e^g S + beta(v - S k) k^T against the paged state cache, the qk l2 "
  "norms, the q readout and the gated rms norm in one launch; one block per (sequence, v head), "
  "the transposed f32 state in one-column-per-thread register arrays, every token walking the "
  "oracle's order",
  "head_k 128 (register bound, as on chunk_scan), head_v up to 128; state_form both forms -- "
  "per_candidate walks the per-token slots, anchor folds the accepted prefix from libr4d's "
  "factor-block layout (six state_idx columns, the stamped window), and the constraint says so; the "
  "state cache's slot/head/column strides are read off the tensor (a paged cache is page-"
  "padded); state_idx rows may be wider than one column (the accepted-prefix walk clamps to the "
  "last), a negative input slot skips the (sequence, head) row whole and a negative output slot "
  "skips only that token's state write, both as the reference's guards do; the o_q/o_scale "
  "quantiser pair writes ref's per-(token,head) e4m3 group epilogue when both are present",
  "f32/bf16 q/k/v/a/b/o and z, f32-or-bf16 A_log/dt_bias/norm_w read as stored (plain single-"
  "plane weights -- the 27B container stores them bf16 and the oracle's dtype-driven ldt reads "
  "the same bytes); the delta-rule chains, the serial-order rms and qk reductions, the double-"
  "rounded exp/softplus/sigmoid and the three-step gated norm are bit-exact by construction, "
  "the scalar sibling of libr4d's recurrent row",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cRecurrent, 4, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_gdn_recurrent_update },

{ "gdn_conv_prep_bf16", "gdn_conv_prep", "gdn",
  "the prefill preamble: depthwise conv with silu and its state commit, the q/k/v split, the "
  "per-head l2 norms, the gate cumsum and beta, in three launches (conv, norm, gate) that keep "
  "the oracle's phase order -- the norms read what the conv stored, as the oracle's re-read does",
  "chunk 64 (the derived c64, the cumsum's reset unit), head_k within the staged-row LDS "
  "budget, conv_width up to the history bound; a (head, sequence, chunk) block walks the "
  "oracle's serial cumsum; absent a/b_gate the gates are the last 2*head_v columns of the "
  "fused x row, and cache_idx/has_init choose the state slot and its cold-window zeros",
  "f32/bf16 x/q/k/v/g/beta with optional b/A_log/dt_bias read as stored (plain single-plane "
  "weights); the conv taps, the serial l2 chains, the chunk-reset cumsum and the double-"
  "rounded silu/sigmoid/softplus all walk the oracle's order -- bit-exact by construction, "
  "the prefill sibling of conv_update",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cConvPrep, 5, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_gdn_conv_prep },

{ "gdn_conv_recurrent_update_bf16", "gdn_conv_recurrent_update", "gdn",
  "the decode pair fused: conv_update into an LDS staging buffer, then the recurrent delta "
  "rule over it, one launch per (sequence, v head); the conv window commits before the "
  "state-idx skip, in the oracle's order -- the pair commits the conv state even when the "
  "recurrent leg then skips the row",
  "the recurrent row's scope exactly: head_k 128, head_v up to 128, both state_form shapes, the "
  "o_q/o_scale quantiser pair honoured (ref's e4m3 group epilogue); conv_width up to 4 (the "
  "window is register-resident); the staging smem follows the conv contract's own sequence "
  "bound, state_len - conv_width + 2",
  "bit-exact against both oracles by the same construction as the two rows it replaces, "
  "because it IS their code paths fused: the staging rounds through x's dtype, so the "
  "round trip the two-op path pays through global memory happens identically here",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cConvFused, 6, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_gdn_conv_recurrent_update },

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

{ "rmsnorm_quant_fp8", "rmsnorm_quant_fp8", "quant",
  "RMS norm and the fp8 group quantiser in one pass; the bf16 norm output is written and "
  "the codes are taken over it, exactly as ref stores the narrowed values before reading "
  "them; with a residual the bf16-narrowed sum is stored before the variance reads it",
  "n a multiple of 4, group 128; one workgroup per row, the row staged in LDS (<= 30720)",
  "bf16 in, fp8_e4m3 codes with an f32 per-group scale, optional bf16/f32 out_bf16",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cRmsNormQuantFp8, 3, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_rmsnorm_quant_fp8 },

{ "dflash_conv_t2_g16_bf16", "dflash_conv", "dflash",
  "DFlash2 grouped dynamic depthwise conv: coefficient add, both taps and the "
  "block-position mask in one pass, one thread per (t, h)",
  "taps 2, group_size 16, hidden_size divisible by 128; delta is a slice of the "
  "[T, 2, taps, NG] projection read at its own row pitch (rank 3 at its own strides "
  "also served); base and out at their own row pitches; out must not alias x",
  "bf16 x / delta / base / out, fp32 accumulate in the oracle's tap order -- "
  "bit-exact by construction under -ffp-contract=off",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cDflashConv, 4, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_dflash_conv },

{ "dflash_select_bf16", "dflash_select", "dflash",
  "DFlash2 candidate selector: the low-rank edge scores and the greedy walk over "
  "them, one workgroup a sequence and no launch between steps",
  "top_k <= 32 and rank a multiple of 32 and <= 512, both being LDS extents; "
  "cand/unary/hp share the row numbering s*steps+l; M derives from numel(cand)/"
  "(steps*top_k) with the parameter as a ceiling; anchor is a dense rank-1 i32 "
  "plane read at anchor_stride",
  "i32 candidate ids, f32 unary logits, bf16 hp and both codebooks, i32 tokens "
  "out; every f32 product and ascending-i sum is the oracle's own chain -- "
  "bit-exact by construction",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cDflashSelect, 6, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_dflash_select },

{ "logit_rerank_bf16", "logit_rerank", "select",
  "R candidates a row rescored exactly against the full-precision head; one wave "
  "per row, thread r owning candidate r's whole chain, then the oracle's "
  "selection sort",
  "any M, R <= 32; plain dense bf16 or f32 [n_vocab, n_embd] head (a joined e4m3 "
  "head declines this hookless row at declare time, as for logits_gemm_bf16); M "
  "derives from numel(x)/n_embd; idx_in/idx/val at their own row pitches, pairs "
  "only densely packed (ref reads it flat)",
  "bf16/f32 x and head, f32 values + i32 ids out, optional f32 pairs plane; the "
  "dots accumulate in DOUBLE ascending k (ref_common.h's named exception) and "
  "narrow once -- bit-exact by construction, the double-chain sibling of libr4d's "
  "fp8-plane row",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cLogitRerank, 6, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_logit_rerank },

{ "logit_rerank_e4m3", "logit_rerank", "select",
  "R candidates a row rescored exactly against the joined e4m3 lm_head; one wave "
  "per row, thread r owning candidate r's whole chain over codes*scale read "
  "element by element, then the oracle's selection sort",
  "any M, R <= 32; the joined e4m3 head r3d_layout_head admits -- codes [n_vocab, "
  "n_embd] e4m3 dense + one float scale per (1 or 128 rows, 128 cols), untrans"
  "formed affine; plain and i8 heads resolve elsewhere by the hook's own answers; "
  "M derives from numel(x)/n_embd; idx_in/idx/val at their own row pitches, pairs "
  "only densely packed (ref reads it flat)",
  "bf16/f32 x, joined e4m3+bf16/f16/f32 head, f32 values + i32 ids out, optional "
  "f32 pairs plane; code*scale is ONE f32 rounding (ref_planes.h ScaledRows), the "
  "dots accumulate in DOUBLE ascending k (ref_common.h's named exception) and "
  "narrow once -- bit-exact by construction, the exact-chain sibling of "
  "logits_gemm_e4m3's bandwidth f32 chain",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cLogitRerank, 6, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_logit_rerank_e4m3,
  nullptr, r3d_layout_head, r3d_relayout_head, nullptr, r3d_unrelayout_head },

{ "sample_ngram_ban_f32", "sample_ngram_ban", "sample",
  "the windowed no-repeat-ngram ban (Unlimited-OCR's "
  "SlidingWindowNoRepeatNgramProcessor): every token that would complete an n-gram "
  "the last `window` tokens of the row's history contain is set to -infinity",
  "any vocabulary; ONE BLOCK A ROW, the search positions dealt across its threads, "
  "the n-1-token prefix staged in LDS (plane reads past 256); integer comparisons "
  "and one constant store, so it is bit-identical to libref, and the same column "
  "written by two positions is an idempotent -infinity; history is a dense i32 "
  "[M, hmax] with numel == M * hmax, the pair plane is dense pairs or [rows, 2] at "
  "its pitch, the whitelist [rows, ceil(n_vocab/32)] at unit column stride",
  "f32 logits in place, i32 history, i32/u32 {n, window} pair, optional i32/u32 "
  "whitelist bitmask, per-row params",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cSampleVoc, 1, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_sample_ngram_ban },

};

/* R3D_NO_GDN=1 withdraws the five GDN rows/schemas from the registration, which is the
 * serve's A/B lever for "GDN arithmetic on the host" without rebuilding a second binary:
 * the same .so, one env difference between arms. The band planner then places those bands
 * on the host exactly as it does for gdn_conv_prep today (they have weights), so startup
 * still completes. Unset is untouched behaviour. */
static bool r3d_gdn_hidden(void) { static const int v = []{ return getenv("R3D_NO_GDN") ? 1 : 0; }(); return v != 0; }
static bool r3d_row_hidden(const char* op) {
    if (!r3d_gdn_hidden() || !op) return false;
    static const char* const kGdn[] = { "gdn_gated_rmsnorm", "gdn_conv_update", "gdn_kkt_solve",
                                        "gdn_chunk_scan", "gdn_recurrent_update",
                                        "gdn_conv_prep", "gdn_conv_recurrent_update" };
    for (const char* n : kGdn) if (strcmp(op, n) == 0) return true;
    return false;
}

/* The decode-band levers, at the same registration gate as R3D_NO_GDN (a launch-time
 * refusal is an abort, not a re-resolve: the engine picks the row at declare and "a
 * kernel that refuses at issue was selected wrongly" -- what R3D_NO_FP8A8=1 would have
 * hit on its first serve A/B, and what R3D_NO_FP8_DEC=1 hit on its first one for real).
 * R3D_NO_FP8A8=1 withdraws all three fp8a8 rows (the bands then place on the host, as
 * they did before the rows shipped); R3D_NO_FP8_DEC=1 withdraws only the decode row, so
 * the M <= 8 band falls to the m64 tile on the same .so -- the issue #4 A/B arm.
 * R3D_NO_FP4_DEC=1 is that same lever for the mxfp4 band (issue #5): since the 2026-10-25
 * q8x promotion it withdraws BOTH width-row legs (legacy walk and q8x sibling) so the
 * band lands whole on the m64 tile -- a lever that hid one leg and left the sibling
 * registered would hand the A/B arm an entry whose backstop refuses (issue #4 class).
 * R3D_MXFP4_Q8X default-ON law: the dot4 q8x arm is its own row
 * (gemm_nt_q_mxfp4_decode_q8x, priority 1200); the literal R3D_MXFP4_Q8X=0 withdraws
 * only that sibling at this gate, and the band resolves to the legacy decode row --
 * byte-identical pre-arm behaviour. R3D_NO_MXFP4=1
 * withdraws all three mxfp4 rows: it was the pre-registration-gdn lever and carried the
 * same latent abort fault #4 caught -- the entry-level refusal now only backstops the gate.
 * The op schemas stay declared either way: gemm_nt_q/gated keep their other-encoding
 * rows. */
static bool r3d_band_row_hidden(const RadKernelInfo& k) {
    static const bool no_fp8  = getenv("R3D_NO_FP8A8") != nullptr;
    static const bool no_f8d  = getenv("R3D_NO_FP8_DEC") != nullptr;
    static const bool no_mx   = getenv("R3D_NO_MXFP4") != nullptr;
    static const bool no_mxd  = getenv("R3D_NO_FP4_DEC") != nullptr;
    static const bool no_df   = getenv("R3D_NO_DFLASH") != nullptr;
    static const bool no_ngb  = getenv("R3D_NO_NGRAM_BAN") != nullptr;
    static const bool q8x_off = r3d_mxfp4_q8x_off();
    if (no_ngb && !strcmp(k.op, "sample_ngram_ban")) return true;   /* engine sampler band */
    if (no_df && (!strcmp(k.op, "dflash_conv") || !strcmp(k.op, "dflash_select") ||
                  !strcmp(k.op, "logit_rerank") ||
                  !strcmp(k.name, "gemm_nt_q_w2a8_decode"))) return true;   /* drafter family (issue #9) */
    if (!strcmp(k.name, "gemm_nt_q_fp8a8_decode")) return no_fp8 || no_f8d;
    if (!strcmp(k.name, "gemm_nt_q_mxfp4_decode")) return no_mx || no_mxd;
    /* the q8x sibling rides every mxfp4-band lever (never leave one leg of a band
     * registered while its family lever hides the other), plus its own OFF lever */
    if (!strcmp(k.name, "gemm_nt_q_mxfp4_decode_q8x")) return no_mx || no_mxd || q8x_off;
    if (!strcmp(k.name, "gemm_nt_q_mxfp4"))        return no_mx;
    return no_fp8 && (!strcmp(k.name, "gemm_nt_q_fp8a8") ||
                      !strcmp(k.name, "gemm_nt_q_gated_fp8a8"));
}

int rad_kernel_schema_count(void) {
    int n = 0;
    for (const RadOpSchema& s : kSchemas) if (!r3d_row_hidden(s.op)) ++n;
    return n;
}
const RadOpSchema* rad_kernel_schema_at(int i) {
    if (i < 0) return nullptr;
    for (const RadOpSchema& s : kSchemas) if (!r3d_row_hidden(s.op) && i-- == 0) return &s;
    return nullptr;
}

/* R3D_NO_ANCHOR=1 narrows both recurrent rows' state_form constraint to per_candidate: the
 * anchor query then resolves to the host reference row while every other band stays on the
 * device. The A/B lever for "the recurrent anchor arithmetic runs on the host" -- the full
 * R3D_NO_GDN lever cannot boot a speculative serve because kkt/chunk_scan have no host
 * placement. Unset is untouched behaviour. */
static void r3d_anchor_gate(void) {
    static const bool once = [] {
        if (getenv("R3D_NO_ANCHOR")) {
            for (RadConstraint& c : cRecurrent)
                if (!strcmp(c.key, "state_form")) c.sval = "per_candidate";
            for (RadConstraint& c : cConvFused)
                if (!strcmp(c.key, "state_form")) c.sval = "per_candidate";
        }
        return true; }();
    (void)once;
}

int rad_kernel_count(void) {
    r3d_anchor_gate();
    int n = 0;
    for (const RadKernelInfo& k : kRows) if (!r3d_row_hidden(k.op) && !r3d_band_row_hidden(k)) ++n;
    return n;
}
const RadKernelInfo* rad_kernel_at(int i) {
    r3d_anchor_gate();
    if (i < 0) return nullptr;
    for (const RadKernelInfo& k : kRows)
        if (!r3d_row_hidden(k.op) && !r3d_band_row_hidden(k) && i-- == 0) return &k;
    return nullptr;
}

} /* extern "C" */
