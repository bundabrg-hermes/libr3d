/* r3d_sample_ngram.hip -- the windowed no-repeat-ngram ban (`sample_ngram_ban`) on RDNA3.
 *
 * Unlimited-OCR's generation_config asks for SlidingWindowNoRepeatNgramProcessor(n=35,
 * window=128): a token that would complete an n-gram the last `window` tokens of the
 * sequence already contain is banned, by setting its logit to -infinity. The oracle is
 * libref's ref_sample_ngram_ban (libref/ref_sample.cpp:267, Apache-2.0, (c) radiance
 * authors -- ported with attribution), and libr4d's row (r4d_sample_ngram_f32.hip) is the
 * provenance for the geometry below; every divergence from it is called out in place.
 *
 * The oracle's rule, with the row's staged history s[ho, ho+hl) oldest first (the
 * committed tokens including the one just emitted), `n` and `w` from the row's i32
 * {n-gram length, window} pair:
 *
 *     search_start = hl - win      (win = w if 0 < w < hl else hl -- w <= 0: the whole span)
 *     search_end   = hl - n + 1
 *     for idx in [search_start, search_end):
 *         if s[idx .. idx + n - 2] == s[hl - n + 1 .. hl - 1):   ban s[idx + n - 1]
 *
 * and at n = 1 the prefix is empty, so every token in the window is banned. The window
 * bounds the n-gram's START index, not its end -- exactly the reference's slice
 * `sequence[search_start:search_end]`, which is why an n-gram spanning the window
 * boundary is still searched.
 *
 * Exactness. Nothing here rounds: the kernel compares int32 history ids and stores the
 * constant -infinity, so its answer is ref_sample_ngram_ban by construction, not by
 * tolerance -- this is the one sampler band with no FP chain to transcribe. The only
 * ordering freedom is that two positions may store -infinity to one column; the store is
 * idempotent and the value is constant. -ffp-contract=off is house law for every device
 * TU and is load-bearing for the file's siblings; here it simply cannot bite.
 *
 * Ids are global and the logits row is this rank's vocabulary shard: a token's column is
 * its id less `vocab_off`, and a token outside the shard is another rank's to mask (it is
 * not a column here at all).
 *
 * The optional `whitelist` plane is the reference's whitelist
 * (modeling_unlimitedocr.py:359,379 -- `banned.difference_update(self.whitelist)` before
 * the writes): word w bit b of the row's own plane row covers column 32w + b, and a SET
 * bit removes that token from the banned set -- it is never written. Absent, nothing is
 * exempt, and the kernel's answer is the one from before the whitelist existed.
 *
 * Refusals (all at resolve, never wrong numbers): logits rank-2 f32 at its row pitch
 * (row2's rule, the file's siblings); history i32, dense [M, hmax] with
 * numel == M * hmax -- ref's hmax is numel/M and a plane whose numel disagrees with
 * M * shape[1] makes ref's own rowoff walk a different view, so it is refused, not
 * guessed (the penalties/dry note, made an equality here); the {n, window} plane is
 * dense pairs (rank 1) or [rows, 2] at its own row pitch with unit column stride, i32/u32
 * -- accepted as ref's ld_int reads them (see the template note at the kernel); the
 * whitelist is [rows, ceil(n_vocab/32)] i32/u32 with unit column stride; M <= 0xffff is
 * the house grid guard every sample row carries.
 */
#include <hip/hip_runtime.h>

#include <cstdio>   /* the launch-proof print, the dflash-family instrument */
#include <cstdlib>  /* getenv: the entry backstop and the debug lever */

#include "rad_abi.h"
#include "rad_plugin.h"
#include "rad_sample.h"   /* RadSampleParams (hist_off/hist_len), the wire layout */

namespace {

constexpr uint32_t kThreads = 256;

/* The prefix staged in LDS, one int32 a token: 1 KiB. A longer n falls back to reading
 * the history plane, which is correct but repeats the prefix's loads per position
 * (libr4d's kMaxPrefix, kept verbatim: n = 35 is what the model card asks for). */
constexpr uint32_t kMaxPrefix = 256;

/* The window both history stages read, restated from r3d_sample.hip's device mirror
 * (anonymous namespace, so no ODR clash between the two TUs), which mirrors
 * ref_sample.cpp's hist_window line for line. last_n = -1 is the ngram ban's "the whole
 * staged span" (ref passes -1); 0 disables, positive keeps the tail. */
__device__ __forceinline__ bool hist_window(const RadSampleParams& sp, int32_t last_n,
                                            int64_t hmax, int64_t* ho_out, int64_t* hl_out) {
    if (last_n == 0) return false;
    int64_t hl = sp.hist_len;
    int64_t ho = sp.hist_off < 0 ? 0 : sp.hist_off;
    if (ho > hmax) ho = hmax;
    if (hl > hmax - ho) hl = hmax - ho;
    if (last_n > 0 && hl > last_n) { ho += hl - last_n; hl = last_n; }
    *ho_out = ho;
    *hl_out = hl;
    return hl > 0;
}

/* One block a row; the search positions dealt across the block's threads. The whole
 * scalar prologue (params row, {n, win} pair, window, bounds) is a pure function of the
 * ROW, so every thread computes it identically -- no shared scalars, no barrier before
 * the prefix staging, and the only __syncthreads is block-uniform, so the divergent-
 * barrier law cannot bite. (libr4d's kernel staged start/end/plen/p0 through thread 0
 * into shared scalars instead; same values, one fewer shared round-trip.)
 *
 * The template parameter is the {n, window} plane's element type: ref reads the pair
 * through ld_int, which sign-extends i32 and ZERO-extends u32 to int64. Reading a u32
 * plane through int32 (what libr4d's kernel does) differs from ref for a pair with the
 * high bit set -- n = 0x80000001 is `n <= 0: skip` under i32 and a huge n (which then
 * fails hl < n and skips too, in practice) under u32. The instantiation mirrors ld_int's
 * rule exactly, so the pair's semantics are ref's for both dtypes. */
template <typename NgT>
__global__ __launch_bounds__(kThreads) void ngram_ban_kernel(
    float* __restrict__ lg, int64_t lg_pitch, int64_t n_vocab,
    const RadSampleParams* __restrict__ p,
    const int32_t* __restrict__ hist, int64_t hist_pitch,
    const NgT* __restrict__ ng, int64_t ng_pitch,
    const uint32_t* __restrict__ wl, int64_t wl_words, int64_t wl_pitch,
    int64_t hmax, int32_t voff) {
    __shared__ int32_t pf[kMaxPrefix];

    const int64_t m = blockIdx.x;
    const RadSampleParams& sp = p[m];
    const int64_t n = (int64_t)ng[m * ng_pitch];
    const int64_t win_req = (int64_t)ng[m * ng_pitch + 1];
    /* Rows whose n is 0 (the default) are untouched: the logits stay bit for bit. */
    if (n <= 0) return;

    /* The whole staged span of the row, clamped to the plane; -1 is ref's "the whole
     * span" for hist_window. `hl < n` is ref's `if (hl < n) continue`. */
    int64_t ho = 0, hl = 0;
    if (!hist_window(sp, -1, hmax, &ho, &hl)) return;
    if (hl < n) return;

    /* w <= 0 (or w >= hl) is the un-windowed processor: the whole staged history. The
     * window bounds the START index, as in the reference. */
    const int64_t win   = (win_req > 0 && win_req < hl) ? win_req : hl;
    const int64_t start = hl - win;
    const int64_t end   = hl - n + 1;
    if (end <= start) return;                  /* ref: `if (search_end <= search_start) continue` */

    /* The prefix s[hl-n+1, hl) (empty at n == 1), read ONCE -- ref's own note: it is the
     * same n-1 tokens compared at every idx. Indices below are relative to ho, as ref's
     * tok_at indexes relative to hb (which carries ho * laststride). */
    const int64_t plen = n - 1;
    const int64_t p0   = hl - plen;

    const int32_t* __restrict__ h  = hist + m * hist_pitch + ho;
    float* __restrict__ row        = lg + m * lg_pitch;
    const uint32_t* __restrict__ w = wl ? wl + m * wl_pitch : nullptr;
    (void)wl_words;   /* bounds: the entry guarantees col >> 5 < wl_words */

    /* The prefix: LDS when it fits, otherwise each position reads it from the plane. */
    const bool in_lds = plen <= (int64_t)kMaxPrefix;
    if (in_lds) {
        for (int64_t j = (int64_t)threadIdx.x; j < plen; j += (int64_t)kThreads)
            pf[j] = h[p0 + j];
        __syncthreads();                        /* block-uniform: reached by every thread */
    }

    for (int64_t idx = start + (int64_t)threadIdx.x; idx < end; idx += (int64_t)kThreads) {
        bool match = true;
        for (int64_t j = 0; j < plen; ++j) {
            const int32_t want = in_lds ? pf[j] : h[p0 + j];
            if (h[idx + j] != want) { match = false; break; }
        }
        if (!match) continue;
        const int32_t tok = h[idx + plen];
        const int64_t col = (int64_t)tok - voff;
        if (tok < 0 || col < 0 || col >= n_vocab) continue;   /* another rank's token */
        /* The whitelist removes the column from the banned set BEFORE the writes (ref's
         * exempt lambda); the raw word is bit-identical for i32 and u32 (the entry casts
         * both to uint32 -- ld_int's sign extension truncates back to the stored bits). */
        if (w && ((w[col >> 5] >> (uint32_t)(col & 31)) & 1u) != 0u) continue;
        /* Two positions may ban one token; the store is idempotent and the value is
         * the constant ref writes: -infinity, seen identically by argmax and every
         * later stage. */
        row[col] = -INFINITY;
    }
}

/* ------------------------------------------------------------------ host-side helpers */
/* The three helpers below are the r3d_sample.hip file-local set, restated for this TU
 * (same rules; anonymous namespace). Params: the byte blob the builder declares as
 * [rows, 24] u32 -- a 32-bit dtype or the row reads wrong, and the byte count must
 * cover M rows (ref's param_rows, same rule). */
static inline bool param_rows_ok(const RadTensor* t, int64_t M) {
    if (!t || !t->data) return false;
    if (t->dtype != RAD_U32 && t->dtype != RAD_I32 && t->dtype != RAD_F32) return false;
    return rad_tensor_numel(t) * 4 >= M * (int64_t)sizeof(RadSampleParams);
}
/* rank-2 row-major-with-pitch reader for the logits row (ref's rowoff/laststride, to
 * the extent our scope can express it: rank 2, unit last stride, pitch = stride[0]). */
static inline int row2(const RadTensor* t, int64_t cols, int64_t* pitch) {
    if (!t || !t->data) return -1;
    if (t->rank != 2) return -2;
    if (t->shape[1] != cols) return -3;
    if (t->stride[1] != 1) return -4;
    *pitch = t->stride[0];
    return 0;
}
/* The history plane: [M, hmax] i32, rows contiguous (stride[0] == shape[1], unit column
 * stride). ref reads any ld_int width and any rowoff walk; the engine's idx planes are
 * i32 dense, and anything else makes ref's numel/M hmax a different view -- refused, not
 * guessed (the penalties/dry note verbatim). The caller additionally pins the entry's
 * numel == M * pitch, which makes this row's hmax EXACTLY ref's numel(hist)/M. */
static int hist_plane(const RadTensor* t, int64_t M, int64_t* pitch) {
    if (!t || !t->data || t->dtype != RAD_I32) return -1;
    if (t->rank != 2 || t->stride[1] != 1 || t->stride[0] != t->shape[1]) return -2;
    if (t->shape[1] <= 0 || rad_tensor_numel(t) < M * t->shape[1]) return -3;
    *pitch = t->stride[0];
    return 0;
}
/* The {n, window} plane: ref's rowoff(ng, m, 2) + laststride rule, to the extent our
 * scope expresses it: a dense rank-1 pair run (reads at 2m / 2m + 1, pitch 2) or a
 * rank-2 [rows, 2] at its own row pitch with unit column stride. At least M pairs
 * either way (ref: "a plane shared by sparser row sets is not read past its end",
 * numel(ng) >= 2M). Pitched rows with shape[1] > 2 are refused (ref's rowoff would
 * then fall back to the flat r*cols view -- a different plane than a pitch reads). */
static int ng_plane(const RadTensor* t, int64_t M, int64_t* pitch) {
    if (!t || !t->data) return -1;
    if (t->rank == 1) {
        if (t->stride[0] != 1) return -2;
        if (rad_tensor_numel(t) < 2 * M) return -3;
        *pitch = 2;
        return 0;
    }
    if (t->rank == 2 && t->shape[1] == 2 && t->stride[1] == 1) {
        if (rad_tensor_numel(t) < 2 * M) return -3;
        if (rad_tensor_numel(t) < (M - 1) * t->stride[0] + 2) return -4;
        *pitch = t->stride[0];
        return 0;
    }
    return -5;
}
/* The whitelist bitmask: ceil(n_vocab/32) words a row, unit column stride, pitched rank-2
 * or dense rank-1, enough words to reach row M-1's last word. i32 or u32 -- the kernel
 * reads raw bits, which is ld_int's answer truncated back for either dtype. */
static int wl_plane(const RadTensor* t, int64_t M, int64_t words, int64_t* pitch) {
    if (!t || !t->data) return -1;
    if (t->dtype != RAD_I32 && t->dtype != RAD_U32) return -2;
    if (t->rank == 1) {
        if (t->stride[0] != 1 || rad_tensor_numel(t) < M * words) return -3;
        *pitch = words;
        return 0;
    }
    if (t->rank == 2 && t->shape[1] == words && t->stride[1] == 1) {
        if (rad_tensor_numel(t) < (M - 1) * t->stride[0] + words) return -4;
        *pitch = t->stride[0];
        return 0;
    }
    return -5;
}

} /* anonymous namespace */

/* ------------------------------------------------------------------ entry */
extern "C" int r3d_rad_sample_ngram_ban(const RadArgs* a, RadStream s) {
    /* Registration-time the row is withdrawn under this env (see the filter in the tail
     * comment); the entry backstop covers a hand-forced launch. Never refuse a shape the
     * constraints accept -- that path is an abort, not a re-resolve. */
    if (getenv("R3D_NO_NGRAM_BAN")) return RAD_E_UNSUPPORTED;

    /* ref: have(a, 4); operands logits, params, history, ngram; optional whitelist. */
    if (a->n_t < 4) return RAD_E_INVAL;
    const RadTensor* lg = rad_arg_in(a, 0);
    const RadTensor* pr = rad_arg_in(a, 1);
    const RadTensor* hi = rad_arg_in(a, 2);
    const RadTensor* ng = rad_arg_in(a, 3);
    const RadTensor* wl = (a->n_t > 4 && a->t[4].data) ? &a->t[4] : nullptr;
    if (!lg || !pr || !hi || !ng) return RAD_E_INVAL;
    if (lg->dtype != RAD_F32) return RAD_E_DTYPE;

    long long nv = 0;
    if (!rad_args_geti(a, "n_vocab", &nv) || nv <= 0) nv = lg->rank >= 2 ? lg->shape[1] : 0;
    if (nv <= 0) return RAD_E_SHAPE;
    const int64_t M = rad_tensor_numel(lg) / nv;
    if (M <= 0 || M > 0xffff) return RAD_E_SHAPE;           /* house grid guard, as siblings */
    if (!param_rows_ok(pr, M)) return RAD_E_SHAPE;
    int64_t lg_p; if (row2(lg, nv, &lg_p)) return RAD_E_STRIDE;
    int64_t hi_p; if (hist_plane(hi, M, &hi_p)) return RAD_E_SHAPE;
    /* ref's hmax is numel(hist)/M; the equality makes shape[1] EXACTLY that, so the
     * kernel's hmax is the oracle's for every accepted plane (see §6.2 of the design). */
    if (rad_tensor_numel(hi) != M * hi_p) return RAD_E_SHAPE;
    if (ng->dtype != RAD_I32 && ng->dtype != RAD_U32) return RAD_E_DTYPE;
    int64_t ng_p; if (ng_plane(ng, M, &ng_p)) return RAD_E_SHAPE;

    const int64_t words = (nv + 31) / 32;
    int64_t wl_p = 0;
    if (wl && wl_plane(wl, M, words, &wl_p)) return RAD_E_STRIDE;

    long long voff = 0; rad_args_geti(a, "vocab_off", &voff);
    const uint32_t* wlp = wl ? (const uint32_t*)wl->data : nullptr;

    /* Two instantiations for the pair dtype -- dry_kernel<BrkT>'s pattern -- so the pair
     * reads carry ld_int's exact sign/zero-extension (see the kernel's template note). */
    if (ng->dtype == RAD_U32)
        hipLaunchKernelGGL(ngram_ban_kernel<uint32_t>, dim3((uint32_t)M), dim3(kThreads), 0,
                           (hipStream_t)(uintptr_t)s, (float*)lg->data, lg_p, nv,
                           (const RadSampleParams*)pr->data, (const int32_t*)hi->data, hi_p,
                           (const uint32_t*)ng->data, ng_p, wlp, words, wl_p, hi_p,
                           (int32_t)voff);
    else
        hipLaunchKernelGGL(ngram_ban_kernel<int32_t>, dim3((uint32_t)M), dim3(kThreads), 0,
                           (hipStream_t)(uintptr_t)s, (float*)lg->data, lg_p, nv,
                           (const RadSampleParams*)pr->data, (const int32_t*)hi->data, hi_p,
                           (const int32_t*)ng->data, ng_p, wlp, words, wl_p, hi_p,
                           (int32_t)voff);
    const hipError_t le = hipGetLastError();
    if (le != hipSuccess) fprintf(stderr, "libr3d: sample_ngram_ban launch FAILED (%s)\n",
                                  hipGetErrorName(le));
    else if (getenv("R3D_DEBUG_NGRAM")) {
        static bool once = false;
        if (!once) { once = true;
            fprintf(stderr, "libr3d: sample_ngram_ban launched (M=%lld n_vocab=%lld "
                            "hmax=%lld whitelist=%d)\n", (long long)M, (long long)nv,
                    (long long)hi_p, wl ? 1 : 0); }
    }
    return le == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* ============================================================================
 * Row-table entry -- paste into src/r3d_rows.cpp (nothing in THIS file declares
 * the row; r3d_rows.cpp is the single declaration TU). Field order is
 * RadKernelInfo's positional order; trailing fields omitted means null, which is
 * this row's truth (no scratch, no layout, no opd_shape -- r4d's row says the same:
 * "the sampler's shapes are declared where the sampler's shapes live").
 *
 * 1) extern, beside the other sample externs (r3d_rows.cpp:1163-1164 block):
 *
 * extern "C" int r3d_rad_sample_ngram_ban(const RadArgs*, RadStream);
 *
 * 2) operand table, verbatim from libr4d (r4d_rows.cpp:2210), beside oDry, and the
 *    schema entry verbatim from libr4d's SCHEMA("sample_ngram_ban", pPenal, oNgramBan,
 *    ...) (r4d_rows.cpp:2952) -- pPenal is libr3d's existing table (r3d_rows.cpp:278,
 *    already shared with sample_penalties/sample_dry). schema_same() must find the same
 *    tables wherever both plugins co-load: a schema disagreement refuses the WHOLE
 *    library at load, so this is load-time correctness, not style. Insert the schema
 *    beside the sample_dry entry (r3d_rows.cpp:786), i.e. after dry and before temp as
 *    libr4d orders them:
 *
 * libr4d's two comment blocks above the table, verbatim from r4d_rows.cpp:2204-2209
 * (carry them into r3d_rows.cpp as block comments, delimiters re-added):
 *
 * "The n-gram ban's own plane: one i32 {n-gram length, window} pair a row (n 0 = off,
 *  window <= 0 = the whole staged history). A plane beside `params` rather than fields
 *  in it, because the params row is the fixed 24-word RadSampleParams every sampler op
 *  shares."
 *
 * "`whitelist` is optional: a bitmask plane a row (a SET bit exempts that column from
 *  the ban, the reference's whitelist), [M, ceil(n_vocab/32)] i32/u32, contiguous.
 *  Absent, nothing is exempt."
 *
 * static const RadOperandSpec oNgramBan[] = {
 *     { "logits",    RAD_OPD_INOUT, 0 },
 *     { "params",    RAD_OPD_IN,    0 },
 *     { "history",   RAD_OPD_IN,    0 },
 *     { "ngram",     RAD_OPD_IN,    0 },
 *     { "whitelist", RAD_OPD_IN,    1 },
 * };
 *
 *      { "sample_ngram_ban", pPenal, (int)(sizeof(pPenal)/sizeof(pPenal[0])),
 *        oNgramBan, (int)(sizeof(oNgramBan)/sizeof(oNgramBan[0])),
 *        "the windowed no-repeat-ngram ban: every token completing an n-gram the last "
 *        "`window` tokens of the row's history contain is set to -infinity. `ngram` is "
 *        "one i32 {n, window} pair a row; the window bounds the n-gram's START index, as "
 *        "in the reference, window <= 0 searches the whole span and n 0 is off. The "
 *        "optional `whitelist` bitmask exempts a token whose column bit is set; absent, "
 *        "nothing is exempt." },
 *
 * 3) constraints: the row reuses the EXISTING cSampleVoc (r3d_rows.cpp:1059, already
 *    libr4d's cSampleVoc = C_GE("n_vocab", 1) spelled as a literal, r4d_rows.cpp:5089).
 *    Nothing new.
 *
 * 4) the row, appended to kRows[] beside the other sample rows (after
 *    sample_dry_f32, r3d_rows.cpp:1511; family "sample", as both libraries spell it):
 *
 * { "sample_ngram_ban_f32", "sample_ngram_ban", "sample",
 *   "the windowed no-repeat-ngram ban (Unlimited-OCR's "
 *   "SlidingWindowNoRepeatNgramProcessor): every token that would complete an n-gram "
 *   "the last `window` tokens of the row's history contain is set to -infinity",
 *   "any vocabulary; ONE BLOCK A ROW, the search positions dealt across its threads, "
 *   "the n-1-token prefix staged in LDS (plane reads past 256); integer comparisons "
 *   "and one constant store, so it is bit-identical to libref, and the same column "
 *   "written by two positions is an idempotent -infinity; history is a dense i32 "
 *   "[M, hmax] with numel == M * hmax, the pair plane is dense pairs or [rows, 2] at "
 *   "its pitch, the whitelist [rows, ceil(n_vocab/32)] at unit column stride",
 *   "f32 logits in place, i32 history, i32/u32 {n, window} pair, optional i32/u32 "
 *   "whitelist bitmask, per-row params",
 *   RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
 *   cSampleVoc, 1, nullptr, 0, nullptr,
 *   nullptr, nullptr, r3d_rad_sample_ngram_ban },
 *
 * 5) kill switch -- R3D_NO_NGRAM_BAN=1 withdraws this row from the registration
 *    enumeration. This is an ENGINE sampler band, not the drafter family, so it gets
 *    its own lever, but it folds into the SAME r3d_band_row_hidden filter
 *    (r3d_rows.cpp:1838) -- that filter is already threaded through both
 *    rad_kernel_count() and rad_kernel_at(), so this is the whole registration edit:
 *
 *    add beside `no_df`:
 *
 * static const bool no_ngb = getenv("R3D_NO_NGRAM_BAN") != nullptr;
 *
 *    and beside the drafter line:
 *
 * if (no_ngb && !strcmp(k.op, "sample_ngram_ban")) return true;   // engine sampler band
 *
 *    A launch-time refusal is an abort, not a re-resolve (the R3D_NO_GDN law,
 *    repeated in every band lever since); withdrawal is the lever, and the entry's
 *    getenv backstop above only covers hand-forced launches. The op SCHEMA stays
 *    declared either way (fp8-lever precedent: declared-schema-without-row is legal,
 *    the reverse is the load error) -- r3d_band_row_hidden filters rows only.
 *
 * 6) CMake: libr3d's CMakeLists lists HIP sources EXPLICITLY (no glob) -- append
 *    src/r3d_sample_ngram.hip to rad_add_plugin's HIP_SOURCES and add the per-unit
 *    flags beside the others (no FP in this op, but the flag pair is house law for
 *    every device TU):
 *
 * set_source_files_properties(src/r3d_sample_ngram.hip PROPERTIES
 *   COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")
 * ========================================================================== */
