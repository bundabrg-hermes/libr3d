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
      oRope, (int)(sizeof(oRope)/sizeof(oRope[0])),
      "rotary embedding in place over the Q and K heads of a fused projection; the V heads "
      "that follow are untouched. The multi-component modes (`mrope`, `imrope`, `axial`) "
      "take [C, M] component-major positions and `sections`; with [M] positions every "
      "component is that one, so mrope and imrope over a text batch are neox exactly." },
};

/* The rmsnorm row. Field order is RadKernelInfo's positional form (rad_abi.h): after the prose,
 * constraints and (absent) tunables come init, fini, launch, scratch, layout, relayout, and
 * opd_shape -- null because a cold-calling tool takes operand shapes from libref's for the op.
 * `replaces` stays zero-initialised: this row is the unfused op, not a fusion. */

static const RadConstraint cBf16Row[] = {
    { "dtype", RAD_C_IN, 0, "bf16" },
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

/* rope: neox and gptj over [M] positions -- a subset of libr4d's five modes. A query for
 * mrope/imrope/axial resolves to no device row on this card, which is an honest failure at
 * issue, not an approximation. */
static const RadConstraint cRopeRow[] = {
    { "head_dim", RAD_C_GE, 1, nullptr },
    { "mode",     RAD_C_IN, 0, "neox gptj" },
};

extern "C" int r3d_rad_rmsnorm(const RadArgs*, RadStream);
extern "C" int r3d_rad_embed_lookup(const RadArgs*, RadStream);
extern "C" int r3d_rad_kv_store(const RadArgs*, RadStream);
extern "C" int r3d_rad_act_mul(const RadArgs*, RadStream);
extern "C" int r3d_rad_cast(const RadArgs*, RadStream);
extern "C" int r3d_rad_rope(const RadArgs*, RadStream);

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
