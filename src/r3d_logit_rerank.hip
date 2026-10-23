/* r3d_logit_rerank.hip -- logit_rerank on RDNA3: R candidate ids per row rescored
 * EXACTLY against the full-precision lm_head, re-sorted on those values.
 *
 * DRAFT for compilation/gating by the owning session -- not yet in CMakeLists, not yet a
 * registered row. Row entry / schema / filter below; derivation in
 * docs (scratch/drafter_rows_design.md).
 *
 * The oracle is libref ref_moe.cpp ref_logit_rerank (Apache-2.0, (c) radiance authors --
 * ported with attribution). Per row m, per candidate r:
 *
 *   g     = idx_in[m][r]                       (an integer read, never through f32)
 *   local = g - vocab_off                      a global id made this rank's column
 *   pad   = g < 0 || local < 0 || local >= n_vocab   ->  (id -1, value -inf)
 *   score = (float) of a DOUBLE accumulator walking k = 0..K-1 ASCENDING:
 *               acc += (double)f(x[m][k]) * (double)f(head[local][k])
 *
 * The DOUBLE chain is not an accuracy luxury: ref_common.h names logit_rerank's dots as
 * one of the documented double-accumulation exceptions -- this op IS the exact
 * rescoring. The device kernel therefore accumulates in f64 (v_*_f64 under
 * -ffp-contract=off, mul and add separately rounded like libref's scalar loop) and
 * narrows ONCE to f32 at the end, reproducing the oracle's chain bit-for-bit. One
 * thread owns one candidate's whole chain: splitting k across lanes would reassociate
 * the double sum and forfeit exactly the property the op exists to provide. RDNA3
 * consumer f64 issue is a fraction of the f32 rate -- accepted here for correctness;
 * the band is tiny (M = sampled positions, R <= 32, one wave per row) and a
 * measurement-gated f32 variant is a follow-up, NOT a launch-time surprise.
 *
 * Re-sort per row: R rounds of selection sort, next_best's total order verbatim --
 * largest value; ties to the LOWER id; pads (id < 0) skipped, so -1 / -inf entries
 * sort last. Identical rule to row_topk / row_topk_merge, which is what makes an
 * in-shard tie and a cross-shard tie resolve alike downstream. The pairs plane (when
 * present) carries the id BIT-PUNNED into an f32 slot followed by its value, exactly as
 * row_topk writes it (memcpy of the int32 == __int_as_float here).
 *
 * Head scope: the PLAIN lm_head -- a dense bf16 or f32 [n_vocab, n_embd] plane. A joined
 * e4m3+scale head (rad_planes header present) is declined: at declare time an encoded
 * band declines this hookless row exactly like logits_gemm_bf16 declines it, and the
 * launcher backstops with rad_planes_header(). The e4m3 logit_rerank row (libr4d's
 * shipping shape: the layout hooks r3d_layout_head / r3d_relayout_head /
 * r3d_unrelayout_head from r3d_gemm_logits.hip already exist for it) is the follow-up
 * named in the design doc; declining at the band is a host placement, never a refusal at
 * issue.
 *
 * M derives from numel(x) / n_embd -- never the schema's M (the arena-capacity rule,
 * r3d_gemm_bf16.hip's account).
 *
 * R3D_NO_DFLASH=1 withdraws this row from REGISTRATION (shared filter snippet lives in
 * r3d_dflash_conv.hip's tail comment; register it once).
 */
#include <hip/hip_runtime.h>
#include "r3d_dump.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rad_abi.h"
#include "rad_plugin.h"   /* rad_arg_in / rad_tensor_numel (host-side entry helpers) */
#include "rad_planes.h"   /* rad_planes_header: the joined-head backstop */

namespace {

constexpr int kMaxR = 32;     /* R <= 32 (constraint): the per-row candidate arrays */

__device__ __forceinline__ float r3d_bf16_to_f32(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}
/* dtype-resolved element read, the r3d_more.hip ldt2 (this row's class is bf16 | f32). */
__device__ __forceinline__ float ldt2(const uint8_t* p, uint32_t dt, int64_t i) {
    return dt == RAD_F32 ? ((const float*)p)[i]
                         : r3d_bf16_to_f32(((const uint16_t*)p)[i]);
}

/* One block (one wave) per row m; thread r owns candidate r's whole double chain, then
 * thread 0 runs next_best's selection sort serially over LDS. A barrier-gated branch on
 * threadIdx only -- never on walk data -- so no divergent barrier arrival. */
__global__ void logit_rerank_kernel(const uint8_t* __restrict__ x, uint32_t xdt,
                                    int64_t x_rs,
                                    const uint8_t* __restrict__ w, uint32_t wdt,
                                    int64_t w_rs,
                                    const int32_t* __restrict__ in, int64_t in_rs,
                                    int32_t* __restrict__ idx, int64_t idx_rs,
                                    float* __restrict__ val, int64_t val_rs,
                                    float* __restrict__ pairs,
                                    int R, int K, int64_t NV, int64_t voff) {
    __shared__ int32_t sid[kMaxR];
    __shared__ float   ssc[kMaxR];

    const int64_t m = blockIdx.x;
    const int r = (int)threadIdx.x;
    if (r < R) {
        const int64_t g = in[m * in_rs + r];
        const int64_t local = g - voff;
        if (g < 0 || local < 0 || local >= NV) {
            sid[r] = -1;
            ssc[r] = -INFINITY;
        } else {
            double acc = 0.0;                                  /* ascending k, no FMA */
            const int64_t xb = m * x_rs, wb = local * w_rs;
            for (int k = 0; k < K; ++k)
                acc += (double)ldt2(x, xdt, xb + k) * (double)ldt2(w, wdt, wb + k);
            sid[r] = (int32_t)g;
            ssc[r] = (float)acc;                               /* the one narrowing */
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        bool used[kMaxR] = { false };
        for (int o = 0; o < R; ++o) {
            int best = -1;
            for (int c = 0; c < R; ++c) {                      /* next_best, verbatim */
                if (used[c] || sid[c] < 0) continue;
                if (best < 0 || ssc[c] > ssc[best] ||
                    (ssc[c] == ssc[best] && sid[c] < sid[best])) best = c;
            }
            const int32_t oid = best >= 0 ? sid[best] : -1;
            const float   ov  = best >= 0 ? ssc[best] : -INFINITY;
            if (best >= 0) used[best] = true;
            idx[m * idx_rs + o] = oid;
            val[m * val_rs + o] = ov;
            if (pairs) {                                       /* (id bits, value) f32 pair */
                float* p = pairs + (m * R + o) * 2;
                p[0] = __int_as_float(oid);
                p[1] = ov;
            }
        }
    }
}

/* Host helpers, the r3d_more.hip pattern. */
static inline bool dt_act_ok(uint32_t dt) { return dt == RAD_F32 || dt == RAD_BF16; }
static inline int64_t param_i(const RadArgs* a, const char* k, int64_t fb) {
    long long v;
    return rad_args_geti(a, k, &v) ? (int64_t)v : fb;
}
static inline int row2(const RadTensor* t, int64_t cols, int64_t* pitch) {
    if (!t || t->rank != 2 || t->stride[1] != 1) return -1;
    *pitch = (t->shape[1] == cols) ? t->stride[0] : cols;
    return 0;
}

} /* anonymous namespace */

/* ------------------------------------------------------------------ entry */
static int r3d_logit_rerank_impl(const RadArgs*, RadStream);
extern "C" int r3d_rad_logit_rerank(const RadArgs* a, RadStream s)
    R3D_DUMP_WRAP("logit_rerank", r3d_logit_rerank_impl)
static int r3d_logit_rerank_impl(const RadArgs* a, RadStream s) {
    if (getenv("R3D_NO_DFLASH")) return RAD_E_UNSUPPORTED;    /* backstop; see filter */

    /* ref: have(a, 5); operands x, head, idx_in, idx, val, pairs? (positional). */
    if (a->n_t < 5) return RAD_E_INVAL;
    const RadTensor* x   = rad_arg_in(a, 0);
    const RadTensor* w   = rad_arg_in(a, 1);
    const RadTensor* in  = rad_arg_in(a, 2);
    const RadTensor* idx = rad_arg_in(a, 3);
    const RadTensor* val = rad_arg_in(a, 4);
    const RadTensor* pairs = (a->n_t > 5 && a->t[5].data) ? &a->t[5] : nullptr;
    if (!x || !w || !in || !idx || !val) return RAD_E_INVAL;

    const int64_t R    = param_i(a, "R", 1);
    const int64_t K    = param_i(a, "n_embd", 0);
    const int64_t NV   = param_i(a, "n_vocab", 0);
    const int64_t voff = param_i(a, "vocab_off", 0);
    if (R <= 0 || K <= 0 || NV <= 0) return RAD_E_SHAPE;
    if (R > kMaxR) return RAD_E_UNSUPPORTED;                  /* the constraint's band */

    /* M from the operand (arena-capacity rule; ref does the same: numel(x) / K). */
    const int64_t M = rad_tensor_numel(x) / K;
    if (M <= 0 || M > 0xffff) return RAD_E_SHAPE;
    if (rad_tensor_numel(in)  < M * R) return RAD_E_SHAPE;
    if (rad_tensor_numel(idx) < M * R) return RAD_E_SHAPE;
    if (rad_tensor_numel(val) < M * R) return RAD_E_SHAPE;

    /* dtypes: this row reads a PLAIN head. A joined (codes+scale) head carries the
     * rad_planes header -- the encoded band declines a hookless row at declare time
     * anyway (the logits_gemm_bf16 precedent); this is the launch backstop. */
    if (!dt_act_ok(x->dtype)) return RAD_E_DTYPE;
    if (!dt_act_ok(w->dtype)) return RAD_E_DTYPE;
    if (rad_planes_header(w->data) != nullptr) return RAD_E_UNSUPPORTED;
    if (in->dtype != RAD_I32 || idx->dtype != RAD_I32 || val->dtype != RAD_F32)
        return RAD_E_DTYPE;
    if (rad_tensor_numel(w) < NV * K) return RAD_E_SHAPE;
    /* The pair plane is f32 words, ids carried by bit pattern, as row_topk writes it. */
    if (pairs && (pairs->dtype != RAD_F32 || rad_tensor_numel(pairs) < M * R * 2))
        return RAD_E_SHAPE;

    int64_t xp = 0, wp = 0, ip = 0, op = 0, vp = 0;
    if (row2(x,   K, &xp)) return RAD_E_STRIDE;
    if (row2(w,   K, &wp)) return RAD_E_STRIDE;
    if (row2(in,  R, &ip)) return RAD_E_STRIDE;
    if (row2(idx, R, &op)) return RAD_E_STRIDE;
    if (row2(val, R, &vp)) return RAD_E_STRIDE;
    /* ref indexes `pairs` FLAT at (m*R+o)*2 -- only a densely-packed plane can honour
     * that, so the checks below enforce the packing and the kernel takes the bare base.
     * The fixture spells rank 3 [M, R, 2]; a flat rank-2 [M, 2R] is the same memory
     * (the r3d_row_topk rule). */
    if (pairs) {
        if (pairs->rank == 3) {
            if (pairs->stride[2] != 1 || pairs->stride[1] != 2 ||
                pairs->stride[0] != 2 * R) return RAD_E_STRIDE;
        } else if (pairs->rank != 2 || pairs->stride[1] != 1 ||
                   pairs->stride[0] < 2 * R) return RAD_E_STRIDE;
    }

    logit_rerank_kernel<<<dim3((unsigned)M), dim3(kMaxR), 0, (hipStream_t)(uintptr_t)s>>>(
        (const uint8_t*)x->data, x->dtype, xp,
        (const uint8_t*)w->data, w->dtype, wp,
        (const int32_t*)in->data, ip,
        (int32_t*)idx->data, op, (float*)val->data, vp,
        pairs ? (float*)pairs->data : nullptr,
        (int)R, (int)K, NV, voff);
    const hipError_t le = hipGetLastError();
    if (le != hipSuccess) fprintf(stderr, "libr3d: logit_rerank launch FAILED (%s)\n",
                                  hipGetErrorName(le));
    else if (getenv("R3D_DEBUG_DFLASH")) {
        static bool once = false;
        if (!once) { once = true;
            fprintf(stderr, "libr3d: logit_rerank launched (M=%lld R=%lld K=%lld NV=%lld "
                            "voff=%lld)\n", (long long)M, (long long)R, (long long)K,
                    (long long)NV, (long long)voff); }
    }
    return le == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* ============================================================================
 * Row-table entry -- paste into src/r3d_rows.cpp. Register the shared
 * R3D_NO_DFLASH registration filter ONCE (its text is in r3d_dflash_conv.hip's
 * tail comment; it already names "logit_rerank" among the three ops).
 *
 * 1) extern:
 *
 * extern "C" int r3d_rad_logit_rerank(const RadArgs*, RadStream);
 *
 * 2) schemas, verbatim libr4d's pLogitRerank/oLogitRerank:
 *
 * static const RadParamSpec pLogitRerank[] = {
 *     { "M",         RAD_P_INT, RAD_REQUIRED },
 *     { "R",         RAD_P_INT, RAD_REQUIRED },
 *     { "n_vocab",   RAD_P_INT, RAD_REQUIRED },
 *     { "n_embd",    RAD_P_INT, RAD_REQUIRED },
 *     { "vocab_off", RAD_P_INT, RAD_OPTIONAL },
 *     { "dtype",     RAD_P_STR, RAD_REQUIRED },
 * };
 * static const RadOperandSpec oLogitRerank[] = {
 *     { "x",      RAD_OPD_IN,     0 },
 *     { "head",   RAD_OPD_WEIGHT, 0 },
 *     { "idx_in", RAD_OPD_IN,     0 },
 *     { "idx",    RAD_OPD_OUT,    0 },
 *     { "val",    RAD_OPD_OUT,    0 },
 *     { "pairs",  RAD_OPD_OUT,    1 },
 * };
 *
 *    and inside kSchemas[]:
 *
 *     { "logit_rerank", pLogitRerank,
 *       (int)(sizeof(pLogitRerank)/sizeof(pLogitRerank[0])),
 *       oLogitRerank, (int)(sizeof(oLogitRerank)/sizeof(oLogitRerank[0])),
 *       "R candidate ids a row rescored EXACTLY against the full-precision head and "
 *       "re-sorted on those values -- the second half of a coarse top-R. `idx_in` "
 *       "carries global ids as row_topk reports them, `vocab_off` is this rank's first "
 *       "column, and an id outside its rows becomes a pad. Ties break on the LOWER id, "
 *       "the rule row_topk and row_topk_merge use, so a tie inside a shard and one "
 *       "between shards resolve alike." },
 *
 * 3) constraints, verbatim libr4d's cLogitRerank (the dtype key names the ACTIVATION
 *    dtype; the head's encoding is decided by the layout hook / the operand's own
 *    dtype, exactly as for the two logits_gemm rows):
 *
 * static const RadConstraint cLogitRerank[] = {
 *     { "M",       RAD_C_GE, 1,   nullptr },
 *     { "R",       RAD_C_GE, 1,   nullptr },
 *     { "R",       RAD_C_LE, 32,  nullptr },
 *     { "n_vocab", RAD_C_GE, 1,   nullptr },
 *     { "n_embd",  RAD_C_GE, 1,   nullptr },
 *     { "dtype",   RAD_C_IN, 0,   "bf16" },
 * };
 *
 * 4) the row, appended to kRows[]. Declared as the ONLY logit_rerank row for now: an
 *    e4m3-head query carries an encoding the resolver asks this hookless row about,
 *    it declines at declare time, and the band lands on the host -- a coverage gap
 *    the follow-up row (with r3d_layout_head / r3d_relayout_head /
 *    r3d_unrelayout_head, declared AFTER this one) closes.
 *
 * { "logit_rerank_bf16", "logit_rerank", "select",
 *   "R candidates a row rescored exactly against the full-precision head; one wave "
 *   "per row, thread r owning candidate r's whole chain, then the oracle's "
 *   "selection sort",
 *   "any M, R <= 32; plain dense bf16 or f32 [n_vocab, n_embd] head (a joined e4m3 "
 *   "head declines this hookless row at declare time, as for logits_gemm_bf16); M "
 *   "derives from numel(x)/n_embd; idx_in/idx/val at their own row pitches, pairs "
 *   "only densely packed (ref reads it flat)",
 *   "bf16/f32 x and head, f32 values + i32 ids out, optional f32 pairs plane; the "
 *   "dots accumulate in DOUBLE ascending k (ref_common.h's named exception) and "
 *   "narrow once -- bit-exact by construction, the double-chain sibling of libr4d's "
 *   "fp8-plane row",
 *   RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
 *   cLogitRerank, 6, nullptr, 0, nullptr,
 *   nullptr, nullptr, r3d_rad_logit_rerank },
 *
 * 5) CMake (no glob; explicit list + per-unit flags):
 *
 * set_source_files_properties(src/r3d_logit_rerank.hip PROPERTIES
 *   COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")
 * ========================================================================== */
