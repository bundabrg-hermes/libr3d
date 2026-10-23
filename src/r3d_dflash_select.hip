/* r3d_dflash_select.hip -- dflash_select on RDNA3: DFlash2's greedy walk over the
 * candidate lattice.
 *
 * DRAFT for compilation/gating by the owning session -- not yet in CMakeLists, not yet a
 * registered row. Row entry / schema / filter below; arithmetic derivation in
 * docs (scratch/drafter_rows_design.md).
 *
 * The oracle is libref ref_quant.cpp ref_dflash_select (Apache-2.0, (c) radiance authors
 * -- ported with attribution). A walk over steps is SERIAL in l (the predecessor at l is
 * the pick at l-1), parallel over sequences m; one workgroup per sequence, no launch
 * between steps (libr4d's geometry, its r4d_dflash_select_bf16 row). Per step l of
 * sequence m, with r = m * steps + l:
 *
 *   pid(0) = anchor[m * anchor_stride]          (an integer read, never through f32)
 *   pid(l) = cand[prev r, best c]               the walk's own previous pick
 *
 *   ax[i]  = (pid in [0, n_vocab) ? f32(pred[pid][i]) : 0.0f)
 *            * (hp ? f32(hp[r][i]) : 1.0f)                       for i in [0, R)
 *   score[c] = f32(unary[r][c])                                        (c in [0, K))
 *              + sum over i ASCENDING of ax[i] * f32(succ[tok_c][i])
 *     ... the succ term added only when tok_c = cand[r][c] is in [0, n_vocab).
 *   best   = argmax over c with STRICT >, first c wins ties (ref's `sc > best_s` scan
 *            from best_s = -inf), best_c = 0 when every score is -inf.
 *
 * Every product and sum is f32 with its own rounding (no FMA: the plugin builds with
 * -ffp-contract=off), bf16->f32 widening is exact, and the ax vector is computed ONCE per
 * step and read K times exactly as ref materialises it -- so the device chain is the
 * oracle's chain instruction-for-instruction and is bit-exact by construction. The guard
 * arithmetic is verbatim: the in-range test zeros the PRED side only, hp still multiplies
 * (0 * hp is part of the contract, including its sign), and a candidate outside the
 * vocabulary scores unary alone -- no succ read.
 *
 * Geometry (constraints verbatim libr4d's cDflashSelect): top_k <= 32 and rank a multiple
 * of 32, <= 512 -- both are LDS extents here (ax[512] f32 = 2 KB, scores[32] f32), and
 * matching libr4d's band keeps drafter fixtures portable between the libraries. steps is
 * the checkpoint's trained block length; it is walked serially inside the block, two
 * barriers per step (stage ax; publish scores), the pick folded through one LDS scalar.
 *
 * M is read from the OPERAND, never the schema's M parameter: rows = numel(cand) /
 * (steps * K) gives M directly (cand carries one row per (sequence, step)), and the
 * parameter -- which the engine may deliver as a RAD_P_RANGE band -- is only a ceiling
 * (ref's band_rows: a call above the band is a caller bug, refused). This is the
 * arena-capacity rule from r3d_gemm_bf16.hip: the parameter is what the selector matched,
 * not this issue's extent.
 *
 * R3D_NO_DFLASH=1 withdraws this row from REGISTRATION (filter snippet at the bottom,
 * shared with r3d_dflash_conv.hip and r3d_logit_rerank.hip -- register the filter once).
 */
#include <hip/hip_runtime.h>
#include "r3d_dump.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rad_abi.h"
#include "rad_plugin.h"   /* rad_arg_in / rad_tensor_numel (host-side entry helpers) */

namespace {

constexpr uint32_t kThreads = 256;
constexpr int kMaxRank = 512;   /* rank <= 512, rank % 32 == 0 (constraint): the ax extent */
constexpr int kMaxTopK = 32;    /* top_k <= 32 (constraint): the scores extent */

__device__ __forceinline__ float r3d_bf16_to_f32(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}

/* One block per sequence m. Threads cooperate on ax (thread-strided over i < rank),
 * lanes 0..top_k-1 score the K edges, and thread 0 performs the serial argmax scan --
 * keeping ref's exact tie behaviour (first strictly-largest c wins) without any shuffle
 * tree that could re-order the comparison. steps x (stage, score, pick): barriers are
 * block-uniform (guarded by nothing but threadIdx, never by walk data), so no divergent
 * barrier arrival can strand the block. */
__global__ void dflash_select_kernel(const int32_t* __restrict__ cand,
                                     const float*   __restrict__ un,
                                     const uint16_t* __restrict__ hp,   /* null = ones */
                                     const int32_t* __restrict__ anc,
                                     const uint16_t* __restrict__ pred,
                                     const uint16_t* __restrict__ succ,
                                     int32_t* __restrict__ tok,
                                     int64_t cand_rs, int64_t un_rs, int64_t hp_rs,
                                     int64_t pred_rs, int64_t succ_rs, int64_t tok_rs,
                                     int64_t astr, int steps, int K, int R, int64_t nv) {
    __shared__ float   ax[kMaxRank];
    __shared__ float   sco[kMaxTopK];
    __shared__ int64_t spid;

    const int64_t m = blockIdx.x;
    int64_t pid = anc[m * astr];                 /* ld_int role: the walk's seed */

    for (int l = 0; l < steps; ++l) {
        const int64_t r = m * steps + l;          /* cand/unary/hp share this row numbering */
        const bool ok = pid >= 0 && pid < nv;
        const int64_t pb = ok ? pid * pred_rs : 0;   /* never form an OOB read address */
        for (int i = (int)threadIdx.x; i < R; i += (int)blockDim.x) {
            const float pv = ok ? r3d_bf16_to_f32(pred[pb + i]) : 0.0f;
            ax[i] = pv * (hp ? r3d_bf16_to_f32(hp[r * hp_rs + i]) : 1.0f);
        }
        __syncthreads();
        if ((int)threadIdx.x < K) {
            const int c = (int)threadIdx.x;
            const int32_t tc = cand[r * cand_rs + c];
            float sc = un[r * un_rs + c];
            if (tc >= 0 && tc < nv) {
                const int64_t sb = (int64_t)tc * succ_rs;
                for (int i = 0; i < R; ++i)
                    sc += ax[i] * r3d_bf16_to_f32(succ[sb + i]);    /* ascending i, no FMA */
            }
            sco[c] = sc;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int best = 0;
            float best_s = -INFINITY;
            for (int c = 0; c < K; ++c)                            /* strict >: ties -> lower c */
                if (sco[c] > best_s) { best_s = sco[c]; best = c; }
            const int64_t pick = cand[r * cand_rs + best];
            spid = pick;
            tok[m * tok_rs + l] = (int32_t)pick;
        }
        __syncthreads();
        pid = spid;                                                /* broadcast pick */
    }
}

/* Host helpers, the r3d_more.hip pattern. */
static inline int64_t param_i(const RadArgs* a, const char* k, int64_t fb) {
    long long v;
    return rad_args_geti(a, k, &v) ? (int64_t)v : fb;
}
/* ref's p_int over a parameter that the engine may deliver as a RAD_P_RANGE band: the
 * band's ceiling is what ref reads, and rad_args_geti only sees RAD_P_INT. */
static inline int64_t param_band(const RadArgs* a, const char* k, int64_t fb) {
    if (!a) return fb;
    const RadParam* p = rad_param_find(a->p, a->n_p, k);
    if (!p) return fb;
    if (p->kind == RAD_P_INT)   return p->ival;
    if (p->kind == RAD_P_RANGE) return p->ihi;
    if (p->kind == RAD_P_STR && p->sval) return strtoll(p->sval, nullptr, 10);
    return fb;
}
/* rank-2 [rows, cols] row pitch with the trailing-stride-unit rule (rowoff's walk
 * degenerates to r*pitch exactly in this scope). */
static inline int row2(const RadTensor* t, int64_t cols, int64_t* pitch) {
    if (!t || t->rank != 2 || t->stride[1] != 1) return -1;
    *pitch = (t->shape[1] == cols) ? t->stride[0] : cols;
    return 0;
}

} /* anonymous namespace */

/* ------------------------------------------------------------------ entry */
static int r3d_dflash_select_impl(const RadArgs*, RadStream);
extern "C" int r3d_rad_dflash_select(const RadArgs* a, RadStream s)
    R3D_DUMP_WRAP("dflash_select", r3d_dflash_select_impl)
static int r3d_dflash_select_impl(const RadArgs* a, RadStream s) {
    if (getenv("R3D_NO_DFLASH")) return RAD_E_UNSUPPORTED;   /* backstop; see filter below */

    /* ref: a->n_t >= 7; positional per the schema: cand(0), unary(1), hp(2, optional),
     * anchor(3), pred(4), succ(5), tokens(6). */
    if (a->n_t < 7) return RAD_E_INVAL;
    const RadTensor* cand = rad_arg_in(a, 0);
    const RadTensor* un   = rad_arg_in(a, 1);
    const RadTensor* hp   = rad_arg_in(a, 2);                 /* absent is a plane of ones */
    const RadTensor* anc  = rad_arg_in(a, 3);
    const RadTensor* pred = rad_arg_in(a, 4);
    const RadTensor* succ = rad_arg_in(a, 5);
    const RadTensor* out  = rad_arg_in(a, 6);
    if (!cand || !un || !anc || !pred || !succ || !out) return RAD_E_INVAL;

    const int64_t steps = param_band(a, "steps", 0);
    const int64_t K     = param_band(a, "top_k", 0);
    const int64_t R     = param_band(a, "rank",
                            pred->rank >= 2 ? pred->shape[pred->rank - 1] : 0);
    const int64_t nv    = param_band(a, "n_vocab",
                            pred->rank >= 2 ? pred->shape[pred->rank - 2] : 0);
    const int64_t astr  = param_band(a, "anchor_stride", 1);

    /* M from the operand, the parameter only a ceiling (band_rows verbatim). */
    if (steps <= 0 || K <= 0) return RAD_E_SHAPE;
    const int64_t rows_operand = rad_tensor_numel(cand) / (steps * K);
    const int64_t band         = param_band(a, "M", 0);
    const int64_t M = (band > 0 && rows_operand > band) ? 0 : rows_operand;
    if (M <= 0 || R <= 0 || nv <= 0 || astr <= 0) return RAD_E_SHAPE;
    if (K > kMaxTopK || R > kMaxRank || (R % 32)) return RAD_E_UNSUPPORTED;  /* LDS extents */

    /* ref's size guards, in ref's order. */
    if (rad_tensor_numel(un)  < M * steps * K) return RAD_E_SHAPE;
    if (rad_tensor_numel(anc) < (M - 1) * astr + 1) return RAD_E_SHAPE;
    if (rad_tensor_numel(out) < M * steps) return RAD_E_SHAPE;
    if (rad_tensor_numel(pred) < nv * R || rad_tensor_numel(succ) < nv * R) return RAD_E_SHAPE;
    if (hp && rad_tensor_numel(hp) < M * steps * R) return RAD_E_SHAPE;

    /* dtypes: i32 ids read with ld_int semantics, f32 unary, bf16 codebooks (the schema
     * spellings); anything else stays on the host row. */
    if (cand->dtype != RAD_I32 || un->dtype != RAD_F32 || out->dtype != RAD_I32 ||
        anc->dtype != RAD_I32 || pred->dtype != RAD_BF16 || succ->dtype != RAD_BF16 ||
        (hp && hp->dtype != RAD_BF16)) return RAD_E_DTYPE;

    int64_t cand_rs = 0, un_rs = 0, hp_rs = 0, pred_rs = 0, succ_rs = 0, tok_rs = 0;
    if (row2(cand, K, &cand_rs)) return RAD_E_STRIDE;
    if (row2(un,   K, &un_rs))   return RAD_E_STRIDE;
    if (hp && row2(hp, R, &hp_rs)) return RAD_E_STRIDE;
    if (row2(pred, R, &pred_rs)) return RAD_E_STRIDE;
    if (row2(succ, R, &succ_rs)) return RAD_E_STRIDE;
    /* anchor: offlin over a rank-1 dense plane is the identity; any view beyond that is
     * refused rather than walked (see design doc: arch-block uncertainty). */
    if (anc->rank != 1 || anc->stride[0] != 1) return RAD_E_STRIDE;
    if (out->rank != 2 || out->stride[1] != 1) return RAD_E_STRIDE;
    tok_rs = (out->shape[1] == steps) ? out->stride[0] : steps;

    if (M > 0xffff) return RAD_E_SHAPE;                       /* grid bound, drafter bands are tiny */
    const dim3 grid((unsigned)M);
    dflash_select_kernel<<<grid, dim3(kThreads), 0, (hipStream_t)(uintptr_t)s>>>(
        (const int32_t*)cand->data, (const float*)un->data,
        hp ? (const uint16_t*)hp->data : nullptr,
        (const int32_t*)anc->data, (const uint16_t*)pred->data, (const uint16_t*)succ->data,
        (int32_t*)out->data, cand_rs, un_rs, hp_rs, pred_rs, succ_rs, tok_rs,
        astr, (int)steps, (int)K, (int)R, nv);
    const hipError_t le = hipGetLastError();
    if (le != hipSuccess) fprintf(stderr, "libr3d: dflash_select launch FAILED (%s)\n",
                                  hipGetErrorName(le));
    else if (getenv("R3D_DEBUG_DFLASH")) {
        static bool once = false;
        if (!once) { once = true;
            fprintf(stderr, "libr3d: dflash_select launched (M=%lld steps=%lld K=%lld "
                            "R=%lld nv=%lld hp=%d)\n", (long long)M, (long long)steps,
                    (long long)K, (long long)R, (long long)nv, hp ? 1 : 0); }
    }
    return le == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* ============================================================================
 * Row-table entry -- paste into src/r3d_rows.cpp (shared registration notes with
 * r3d_dflash_conv.hip; register the R3D_NO_DFLASH filter once).
 *
 * 1) extern:
 *
 * extern "C" int r3d_rad_dflash_select(const RadArgs*, RadStream);
 *
 * 2) schemas, verbatim libr4d's pDflashSelect/oDflashSelect:
 *
 * static const RadParamSpec pDflashSelect[] = {
 *     { "M",       RAD_P_INT, RAD_REQUIRED },
 *     { "steps",   RAD_P_INT, RAD_REQUIRED },
 *     { "top_k",   RAD_P_INT, RAD_REQUIRED },
 *     { "rank",    RAD_P_INT, RAD_REQUIRED },
 *     { "n_vocab", RAD_P_INT, RAD_REQUIRED },
 *     { "anchor_stride", RAD_P_INT, RAD_OPTIONAL },
 *     { "dtype",   RAD_P_STR, RAD_REQUIRED },
 * };
 * static const RadOperandSpec oDflashSelect[] = {
 *     { "cand",   RAD_OPD_IN,     0 },
 *     { "unary",  RAD_OPD_IN,     0 },
 *     { "hp",     RAD_OPD_IN,     1 },
 *     { "anchor", RAD_OPD_IN,     0 },
 *     { "pred",   RAD_OPD_WEIGHT, 0 },
 *     { "succ",   RAD_OPD_WEIGHT, 0 },
 *     { "tokens", RAD_OPD_OUT,    0 },
 * };
 *
 *    and inside kSchemas[]:
 *
 *     { "dflash_select", pDflashSelect,
 *       (int)(sizeof(pDflashSelect)/sizeof(pDflashSelect[0])),
 *       oDflashSelect, (int)(sizeof(oDflashSelect)/sizeof(oDflashSelect[0])),
 *       "DFlash2's candidate selector: the low-rank edge score between consecutive draft "
 *       "positions, and the greedy walk that turns a block of independent top-k sets into "
 *       "one path. Only the K edges leaving the previous pick are scored, because the walk "
 *       "is serial and the other K-1 rows of the score matrix are never read." },
 *
 * 3) constraints, verbatim libr4d's cDflashSelect:
 *
 * static const RadConstraint cDflashSelect[] = {
 *     { "steps",   RAD_C_GE, 1,   nullptr },
 *     { "top_k",   RAD_C_GE, 1,   nullptr },
 *     { "top_k",   RAD_C_LE, 32,  nullptr },
 *     { "rank",    RAD_C_DIV, 32, nullptr },
 *     { "rank",    RAD_C_LE, 512, nullptr },
 *     { "dtype",   RAD_C_IN, 0,   "bf16" },
 * };
 *
 * 4) the row, appended to kRows[]:
 *
 * { "dflash_select_bf16", "dflash_select", "dflash",
 *   "DFlash2 candidate selector: the low-rank edge scores and the greedy walk over "
 *   "them, one workgroup a sequence and no launch between steps",
 *   "top_k <= 32 and rank a multiple of 32 and <= 512, both being LDS extents; "
 *   "cand/unary/hp share the row numbering s*steps+l; M derives from numel(cand)/"
 *   "(steps*top_k) with the parameter as a ceiling; anchor is a dense rank-1 i32 "
 *   "plane read at anchor_stride",
 *   "i32 candidate ids, f32 unary logits, bf16 hp and both codebooks, i32 tokens "
 *   "out; every f32 product and ascending-i sum is the oracle's own chain -- "
 *   "bit-exact by construction",
 *   RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
 *   cDflashSelect, 6, nullptr, 0, nullptr,
 *   nullptr, nullptr, r3d_rad_dflash_select },
 *
 * 5) CMake (no glob; explicit list + per-unit flags):
 *
 * set_source_files_properties(src/r3d_dflash_select.hip PROPERTIES
 *   COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")
 * ========================================================================== */
