/* ============================================================================
 * r3d_logit_rerank_e4m3.hip -- logit_rerank on RDNA3 over the JOINED e4m3 lm_head.
 *
 * DRAFT for compilation/gating by the owning session -- not in CMakeLists, not a
 * registered row. Registration snippet in the tail comment; design doc alongside
 * this file (design.md). This file does not touch r3d_logit_rerank.hip: the bf16
 * arm shipped gated bit-exact and stays untouched; this TU is its joined-e4m3
 * sibling, the row that file's header comment names as the follow-up.
 *
 * The band: rad_block_dflash2.h's op_drerank (RAD_WEIGHTS(w_lm)) in the qwen35_fp8
 * arch -- the DFlash2 selector's exact rescore: the 2-bit coarse head proposes its
 * top-R (R = kRerankR = 32) candidate ids, and each is re-scored against the
 * TARGET's lm_head, declared with those scores. w_lm is set by the plugin only when
 * output.weight's codes are RAD_F8E4M3 or RAD_I8 (qwen35_fp8.cpp, declare_dflash2);
 * the fp8 case is the 27B container's joined e4m3 form -- a RadPlanesHeader blob of
 * e4m3 codes [n_vocab, n_embd] (block 1x1) + a bf16/f16/f32 scale per
 * (1-or-128 rows, 128 cols), exactly what r3d_layout_head admits and what this
 * kernel reads. The x operand is sel_h: the gathered final hidden rows, bf16
 * (act_dtype; gather_rows is its only writer -- NOT a quantised pair).
 *
 * The oracle is libref ref_moe.cpp ref_logit_rerank (Apache-2.0, (c) radiance
 * authors -- ported with attribution), joined arm. Per row m, per candidate r:
 *
 *   g     = idx_in[m][r]                        (an integer read, never through f32)
 *   local = g - vocab_off                       a global id made this rank's column
 *   pad   = g < 0 || local < 0 || local >= n_vocab  ->  (id -1, value -inf)
 *   acc (double), k = 0..K-1 ASCENDING, no FMA:
 *       s   = scale[(local / br) * rowscale][k / 128]    widened exactly to f32
 *       hv  = f32(e4m3 code[local][k]) * s               ONE f32 rounding
 *                                                       (ref_planes.h ScaledRows::at;
 *                                                        the e4m3 widen is exact)
 *       acc += (double) f32(x[m][k]) * (double) hv       the double multiply of two
 *                                                        f32s is exact; the ADD is
 *                                                        the double-rounded step
 *   score = (float) acc                                   the one narrowing
 *
 * The DOUBLE chain is not an accuracy luxury: ref_common.h names logit_rerank's dots
 * as the documented double-accumulation exception -- this op IS the exact rescoring.
 * Do NOT port this from r3d_gemm_logits.hip's logits_e4m3 kernel, which folds
 * x * (code * s) into one f32 __fmaf_rn chain and reassociates through a butterfly:
 * that is fine for a bandwidth row kbench adjudicates on rel_l2, and WRONG here,
 * where kbench gates bit-exactness. One thread owns one candidate's whole chain:
 * splitting k across lanes would reassociate the double sum and forfeit exactly the
 * property the op exists to provide. RDNA3 consumer f64 issue is a fraction of the
 * f32 rate -- accepted for correctness; the band is tiny (M = draft positions,
 * R <= 32, one wave per row) and reads ONE byte per product against the bf16 arm's
 * two. A uint4 per-thread codes walk (16 codes per iteration, order-preserving) is
 * the measurement-gated follow-up, never a launch-time surprise.
 *
 * Re-sort per row: R rounds of selection sort, next_best's total order verbatim --
 * largest value; ties to the LOWER id; pads (id < 0) skipped, so -1 / -inf entries
 * sort last. Same rule as row_topk / row_topk_merge, so an in-shard tie and a
 * cross-shard tie resolve alike downstream. The pairs plane (when present) carries
 * the id BIT-PUNNED into an f32 slot followed by its value, exactly as row_topk
 * writes it (memcpy of the int32 == __int_as_float here).
 *
 * Selection contract (the "NEVER issue-refuse" half): this row is declared AFTER
 * logit_rerank_bf16 and carries r3d_layout_head / r3d_relayout_head /
 * r3d_unrelayout_head (r3d_gemm_logits.hip's, already extern'd in r3d_rows.cpp for
 * logits_gemm_e4m3). The hook admits EXACTLY the joined e4m3 codes + one float
 * scale, untransformed, scheme "affine" -- anything else it refuses:
 *   * plain dense head: the hook answers "unsupported" = STORE AS IS (a decline
 *     would steal the plain band -- core/plugin/select.cpp, COVERAGE.md 2026-10-20),
 *     and the earlier bf16 row claims it anyway;
 *   * joined i8 head (the band's other legal w_lm): the hook answers RAD_E_DTYPE --
 *     a declare-time decline, so the band lands on the host row and serve never
 *     aborts. libr4d ships an i8 device row; libr3d's is a separate follow-up.
 * The dtype PARAMS match both rows identically (cLogitRerank, dtype IN "bf16" --
 * the key names the ACTIVATION dtype; the head's form is a property of the weight
 * encoding the hooks see). See the design doc: the qwen35_fp8 band DECLARES this op
 * at dtype "bf16" (geom_from(m.g, meta, ctx, "bf16", RAD_BF16)), so a row gated on
 * dtype "fp8a8" would never match it.
 *
 * The launcher's RAD_E_UNSUPPORTED / RAD_E_DTYPE / RAD_E_SHAPE returns are the same
 * dead-code backstops the bf16 arm ships: the resolver + hook already keep this row
 * off every band it cannot serve; they fire only if that machinery is bypassed
 * (a direct kbench call naming this row).
 *
 * M derives from numel(x) / n_embd -- never the schema's M (the arena-capacity
 * rule, r3d_gemm_bf16.hip's account; the ref does the same).
 *
 * R3D_NO_DFLASH=1 withdraws this row from REGISTRATION automatically: the shared
 * filter (r3d_rows.cpp, r3d_band_row_hidden) matches k.op == "logit_rerank", which
 * covers every row of the op, this one included. No new lever, no filter edit.
 * ========================================================================== */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "r3d_dump.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rad_abi.h"
#include "rad_plugin.h"   /* rad_arg_in / rad_tensor_numel (host-side entry helpers) */
#include "rad_planes.h"   /* the joined form: header, entries, rad_planes_find */

namespace {

constexpr int kMaxR = 32;     /* R <= 32 (cLogitRerank): the per-row candidate arrays */

__device__ __forceinline__ float r3d_bf16_to_f32(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}
/* dtype-resolved element read of x -- the r3d_more.hip ldt2 (this arm's x stays
 * bf16 | f32 exactly as in the plain-head arm: the joined form is the HEAD's). */
__device__ __forceinline__ float ldt2(const uint8_t* p, uint32_t dt, int64_t i) {
    return dt == RAD_F32 ? ((const float*)p)[i]
                         : r3d_bf16_to_f32(((const uint16_t*)p)[i]);
}

/* rad_plugin.h rad_fp8e4m3_to_f32, ported verbatim (r3d_gemm_logits.hip's copy, same
 * rationale: one place for the spec to live). Every code widens EXACTLY, the oracle's
 * rad_load_f32 over e4m3 included (subnormals as m * 2^-9, one NaN form). */
__device__ __forceinline__ float fp8e4m3_m4(uint8_t b) {
    const uint32_t s = (uint32_t)(b >> 7) << 31;
    const uint32_t e = (b >> 3) & 0xFu, m = b & 7u;
    union { uint32_t u; float f; } v;
    if (e == 0) {
        if (m == 0) { v.u = s; return v.f; }
        float f = (float)m / 512.0f;             /* m * 2^-9, exact */
        v.u = s | (__float_as_uint(f) & 0x7fffffffu);
        return v.f;
    }
    if (e == 0xF && m == 7) { v.u = s | 0x7fc00000u; return v.f; }   /* NaN */
    v.u = s | ((e - 7u + 127u) << 23) | (m << 20);
    return v.f;
}

/* One scale element of the plane's dtype, widened exactly -- ref_planes.h
 * JPlane::at over the scale plane (bf2f / __half2float / identity, all exact). */
__device__ __forceinline__ float ld_scale(const void* base, int64_t idx, uint32_t dt) {
    if (dt == RAD_F32)  return ((const float*)base)[idx];
    if (dt == RAD_F16)  return __half2float(((const __half*)base)[idx]);
    return r3d_bf16_to_f32(((const uint16_t*)base)[idx]);
}

/* One block (one wave) per row m; thread r owns candidate r's whole double chain --
 * its head row `local` is its OWN, so (unlike logits_e4m3, where a whole warp shares
 * one head row) the scale element cannot be hoisted warp-wide: each thread walks its
 * own candidate row and reloads the scale once per 128-column block, the value the
 * oracle's scale.cover(local, k) gives at every k of that block (a hoist of a
 * per-block-constant is value-preserving; no rounding moves). Then thread 0 runs
 * next_best's selection sort serially over LDS. A barrier-gated branch on threadIdx
 * only -- never on walk data -- so no divergent barrier arrival. */
__global__ void logit_rerank_e4m3_kernel(const uint8_t* __restrict__ x, uint32_t xdt,
                                         int64_t x_rs,
                                         const uint8_t* __restrict__ codes, int64_t code_rs,
                                         const void* __restrict__ scale, int64_t scale_rs,
                                         int br, uint32_t sdt,
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
            const int64_t xb  = m * x_rs;
            const uint8_t* crow = codes + local * code_rs;
            const int64_t sbase = (local / br) * scale_rs;     /* scale row of the grid */
            float s = ld_scale(scale, sbase, sdt);             /* k < 128 */
            for (int k = 0; k < K; ++k) {
                if ((k & 127) == 0 && k > 0)                   /* the cover's column block */
                    s = ld_scale(scale, sbase + (k >> 7), sdt);
                const float hv = fp8e4m3_m4(crow[k]) * s;      /* ONE f32 rounding */
                acc += (double)ldt2(x, xdt, xb + k) * (double)hv;
            }
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

/* Host helpers, the r3d_logit_rerank.hip pattern. */
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
static inline int64_t dt_bytes(uint32_t dt) {   /* the three scale dtypes only */
    return dt == RAD_F32 ? 4 : 2;
}

} /* anonymous namespace */

/* ------------------------------------------------------------------ entry */
static int r3d_logit_rerank_e4m3_impl(const RadArgs*, RadStream);
extern "C" int r3d_rad_logit_rerank_e4m3(const RadArgs* a, RadStream s)
    R3D_DUMP_WRAP("logit_rerank_e4m3", r3d_logit_rerank_e4m3_impl)
static int r3d_logit_rerank_e4m3_impl(const RadArgs* a, RadStream s) {
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

    /* x is the band's bf16 sel_h (f32 accepted, as the plain arm). */
    if (!dt_act_ok(x->dtype)) return RAD_E_DTYPE;
    if (in->dtype != RAD_I32 || idx->dtype != RAD_I32 || val->dtype != RAD_F32)
        return RAD_E_DTYPE;

    /* The head must be the joined form the layout hook admitted -- the same plane
     * checks logits_e4m3's launcher makes, widened to ref_planes.h scaled_rows:
     * codes e4m3 1x1 over [NV, K] dense; one float-grid scale. A PLAIN head
     * (no header) is the bf16 row's contract -- reaching here means the resolver
     * was bypassed, so refuse as unsupported, never decode a plain plane here. */
    const RadPlanesHeader* h = rad_planes_header(w->data);
    if (!h || h->n_planes != 2) return h ? RAD_E_DTYPE : RAD_E_UNSUPPORTED;
    if (h->transform[0]) return RAD_E_UNSUPPORTED;
    const RadPlanesEntry* ec = nullptr;
    const RadPlanesEntry* es = nullptr;
    const void* pc = rad_planes_find(w->data, "codes", &ec);
    const void* ps = rad_planes_find(w->data, "scale", &es);
    if (!pc || !ps || !ec || !es) return RAD_E_DTYPE;
    if (ec->dtype != RAD_F8E4M3 || ec->block[0] != 1 || ec->block[1] != 1 ||
        ec->rows != NV || ec->cols != K) return RAD_E_SHAPE;
    if (ec->row_bytes != K) return RAD_E_STRIDE;              /* dense, one byte each */
    if (es->dtype != RAD_BF16 && es->dtype != RAD_F16 && es->dtype != RAD_F32)
        return RAD_E_DTYPE;                                    /* scaled_rows's grid */
    if (es->block[1] != 128 || (es->block[0] != 1 && es->block[0] != 128))
        return RAD_E_DTYPE;
    const int64_t srows = (NV + es->block[0] - 1) / es->block[0];
    const int64_t scols = (K + 127) / 128;
    if (es->rows != srows || es->cols != scols) return RAD_E_SHAPE;
    if ((int64_t)h->bytes < es->offset + es->rows * es->row_bytes) return RAD_E_SHAPE;
    /* The oracle walks the scale at ITS row_bytes; honour that stride exactly
     * (JPlane::at uses row_bytes -- strictly more faithful than logits_e4m3's
     * dense assumption, identical when the plane is dense, which rad_planes.h's
     * `row_bytes = rad_dtype_bytes(dtype, cols)` makes it today). */
    if (es->row_bytes < scols * dt_bytes(es->dtype)) return RAD_E_STRIDE;
    const int64_t scale_rs = es->row_bytes / dt_bytes(es->dtype);

    /* The pair plane is f32 words, ids carried by bit pattern, as row_topk writes
     * it -- ref indexes it FLAT at (m*R+o)*2, so the plane must be densely packed
     * (the r3d_row_topk rule; the fixture spells it rank 3 [M, R, 2]). */
    if (pairs && (pairs->dtype != RAD_F32 || rad_tensor_numel(pairs) < M * R * 2))
        return RAD_E_SHAPE;
    if (pairs) {
        if (pairs->rank == 3) {
            if (pairs->stride[2] != 1 || pairs->stride[1] != 2 ||
                pairs->stride[0] != 2 * R) return RAD_E_STRIDE;
        } else if (pairs->rank != 2 || pairs->stride[1] != 1 ||
                   pairs->stride[0] < 2 * R) return RAD_E_STRIDE;
    }

    int64_t xp = 0, ip = 0, op = 0, vp = 0;
    if (row2(x,   K, &xp)) return RAD_E_STRIDE;
    if (row2(in,  R, &ip)) return RAD_E_STRIDE;
    if (row2(idx, R, &op)) return RAD_E_STRIDE;
    if (row2(val, R, &vp)) return RAD_E_STRIDE;

    logit_rerank_e4m3_kernel<<<dim3((unsigned)M), dim3(kMaxR), 0,
                               (hipStream_t)(uintptr_t)s>>>(
        (const uint8_t*)x->data, x->dtype, xp,
        (const uint8_t*)pc, ec->row_bytes,
        ps, scale_rs, (int)es->block[0], es->dtype,
        (const int32_t*)in->data, ip,
        (int32_t*)idx->data, op, (float*)val->data, vp,
        pairs ? (float*)pairs->data : nullptr,
        (int)R, (int)K, NV, voff);
    const hipError_t le = hipGetLastError();
    if (le != hipSuccess) fprintf(stderr, "libr3d: logit_rerank_e4m3 launch FAILED (%s)\n",
                                  hipGetErrorName(le));
    else if (getenv("R3D_DEBUG_DFLASH")) {
        static bool once = false;
        if (!once) { once = true;
            fprintf(stderr, "libr3d: logit_rerank_e4m3 launched (M=%lld R=%lld K=%lld "
                            "NV=%lld voff=%lld br=%lld sdt=%u)\n", (long long)M,
                    (long long)R, (long long)K, (long long)NV, (long long)voff,
                    (long long)es->block[0], es->dtype); }
    }
    return le == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* ============================================================================
 * Registration -- paste into src/r3d_rows.cpp. NOTHING ELSE changes: no schema
 * (the op's pLogitRerank/oLogitRerank are already declared -- the plugin declares
 * one schema per op name), no constraint table (cLogitRerank is shared verbatim,
 * exactly as libr4d shares it between its fp8 and i8 logit_rerank rows), no kill-
 * switch edit (r3d_band_row_hidden matches k.op == "logit_rerank"), no new hook
 * (r3d_layout_head / r3d_relayout_head / r3d_unrelayout_head are r3d_gemm_logits.hip's,
 * already extern'd beside logits_gemm_e4m3's entry).
 *
 * ORDER IS THE CONTRACT: append AFTER the logit_rerank_bf16 row. Declaring first
 * would steal the plain bf16 band (the layout hook's "unsupported" answer for a
 * single-plane weight means STORE AS IS, not decline -- select.cpp, COVERAGE.md
 * 2026-10-20's logits_gemm_e4m3 lesson), then refuse those launches. Declared
 * second: the plain band resolves to the bf16 row before this one is consulted,
 * and the joined band DECLINES the hookless bf16 row at declare time (multi-plane
 * weights refuse hookless device rows), leaving this row the only claimant. The
 * i8 joined band declines BOTH rows (this row's hook answers RAD_E_DTYPE for i8
 * codes) and lands on the host -- a placement, never an issue-refuse.
 *
 * 1) extern, beside the bf16 arm's:
 *
 * extern "C" int r3d_rad_logit_rerank_e4m3(const RadArgs*, RadStream);
 *
 * 2) the row, appended to kRows[] AFTER logit_rerank_bf16:
 *
 * { "logit_rerank_e4m3", "logit_rerank", "select",
 *   "R candidates a row rescored exactly against the joined e4m3 lm_head; one wave "
 *   "per row, thread r owning candidate r's whole chain over codes*scale read "
 *   "element by element, then the oracle's selection sort",
 *   "any M, R <= 32; the joined e4m3 head r3d_layout_head admits -- codes [n_vocab, "
 *   "n_embd] e4m3 dense + one float scale per (1 or 128 rows, 128 cols), untrans-"
 *   "formed affine; plain and i8 heads resolve elsewhere by the hook's own answers; "
 *   "M derives from numel(x)/n_embd; idx_in/idx/val at their own row pitches, pairs "
 *   "only densely packed (ref reads it flat)",
 *   "bf16/f32 x, joined e4m3+bf16/f16/f32 head, f32 values + i32 ids out, optional "
 *   "f32 pairs plane; code*scale is ONE f32 rounding (ref_planes.h ScaledRows), the "
 *   "dots accumulate in DOUBLE ascending k (ref_common.h's named exception) and "
 *   "narrow once -- bit-exact by construction, the exact-chain sibling of "
 *   "logits_gemm_e4m3's bandwidth f32 chain",
 *   RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
 *   cLogitRerank, 6, nullptr, 0, nullptr,
 *   nullptr, nullptr, r3d_rad_logit_rerank_e4m3,
 *   nullptr, r3d_layout_head, r3d_relayout_head, nullptr, r3d_unrelayout_head },
 *
 *    (RadKernelInfo field order per abi/rad_abi.h: ..., constraints, n_constraints,
 *    tunables, 0, tune_valid, init, fini, launch, scratch, layout, relayout,
 *    opd_shape, unrelayout. No opd_shape for logit_rerank's own reason -- libr4d's
 *    row states it: the declared shape and the stored joined planes differ, so a
 *    buffer sized from the declared shape would be read past its end; kbench sizes
 *    the band's side with libref's ref_logit_rerank_shape, as for logits_gemm_e4m3.)
 *
 * 3) CMake (no glob; explicit list + per-unit flags), both lines beside the bf16
 *    arm's existing ones:
 *
 * set_source_files_properties(src/r3d_logit_rerank_e4m3.hip PROPERTIES
 *   COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")
 *
 *    ... and src/r3d_logit_rerank_e4m3.hip appended to rad_add_plugin's HIP_SOURCES
 *    (the line that already names src/r3d_logit_rerank.hip).
 * ========================================================================== */
