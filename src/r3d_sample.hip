/* r3d_sample.hip -- libr3d's classic sampling chain on RDNA3.
 *
 * The arithmetic is libref's (libref/ref_sample.cpp, Apache-2.0, (c) radiance authors --
 * ported with attribution), and the goal is exact agreement, not approximation: sampler
 * outputs are indices, so a probability that lands on a different side of a drop boundary
 * is a different token, not a small error. Three rules make agreement mechanical:
 *
 *   * order-dependent sums (the softmax normaliser, the nucleus prefix) are accumulated
 *     SEQUENTIALLY in candidate order by one thread. libr4d's own rows say the same thing:
 *     "a parallel prefix sums in a different order and a candidate at the cut would survive
 *     in one and not the other";
 *   * every transcendental is exp(double) rounded once to f32, matching ref's rule that
 *     glibc float routines are not the same function across releases (ref_common.h);
 *   * the candidate order is the total order (value descending, tie to the LOWER id), and
 *     every selection -- argmax, top-k rounds -- is one comparison rule: `r3d_beats`.
 *     topk_next's "strictly after the previous pick" scan is exactly this order, so k
 *     parallel argmax rounds under it reproduce the reference set, descending, tie for tie.
 *
 * RNG: splitmix64 over (seed * golden + pos), 24 high bits as f32 -- the spec stream from
 * abi/rad_sample.h, restated as __device__ functions because the header's inline versions
 * are host-only in this translation-unit layout.
 *
 * This file covers the whole chain the core issues for a sampled request:
 * sample_temp, sample_mask, sample_argmax, sample_topk, sample_topp, sample_minp,
 * sample_pick, and the flag-gated stages sample_penalties, sample_dry,
 * sample_typical, sample_xtc and sample_logprobs. The last five have no host stand-in
 * for the engine (no weights for the planner to place), so the model cannot START
 * without device rows for them -- the serve gate is what demanded this batch.
 *
 * Two refusals remain honest misses: the history/breaker planes must arrive i32 (ref
 * reads any ld_int width; the engine's idx planes are i32), and the DRY window is
 * bounded by what fits LDS (hmax <= 6144).
 *
 * Refusals (all at resolve, never wrong numbers): logits/candidate tensors must be rank 2
 * with a unit last stride; the vocab-parallel `pairs` plane of sample_topk is refused
 * (single-rank scope); params must be the 32-bit byte blob the builder emits; integer
 * planes (cand_idx, token, bitmask) must be the widths the engine sends.
 */
#include <hip/hip_runtime.h>

#include "rad_abi.h"
#include "rad_plugin.h"
#include "rad_sample.h"   /* RadSampleParams, RAD_SP_*, RAD_SAMPLE_KEY_* (wire layout) */

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kWarps   = kThreads / 32;   /* wave32 ABI is pinned by CMake */

/* ------------------------------------------------------------------ spec RNG (abi/rad_sample.h) */
__device__ __forceinline__ uint64_t r3d_splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
__device__ __forceinline__ float r3d_uniform01(uint64_t seed, uint64_t pos) {
    const uint64_t z = r3d_splitmix64(seed * 0x9E3779B97F4A7C15ull + pos);
    return (float)(uint32_t)(z >> 40) * (1.0f / 16777216.0f);
}
/* The pick key comes from the header's own table (RAD_SAMPLE_KEY_PICK). */

/* ref's transcendental rule: evaluate in double, round once (ref_common.h's ref_exp/ref_log). */
__device__ __forceinline__ float r3d_exp(float x) { return (float)exp((double)x); }
__device__ __forceinline__ float r3d_log(float x) { return (float)log((double)x); }

/* The window both history stages read: the row's own row of the history plane, clamped to
 * it, with `last_n` keeping the tail of the span. Mirrors ref_sample.cpp's hist_window
 * line for line: 0 disables the stage, negative keeps the whole span. */
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

/* ------------------------------------------------------------------ shared selection rule */
/* (v desc, id asc): `beats` is the ONLY ordering rule in this file. bi < 0 is the empty
 * slot; the first real element always fills it, NaN or not -- exactly ref's `bi < 0 ||`.
 * Comparison-after-empty means NaN never displaces a real value (both comparisons false). */
__device__ __forceinline__ bool r3d_beats(float av, int64_t ai, float bv, int64_t bi) {
    return bi < 0 || av > bv || (av == bv && ai < bi);
}

/* Block-wide argmax over the row: threads stride `n`, reduce by `beats`. `keep` gates
 * candidates (top-k rounds); pass r3d_keep_all for the plain scan. Returns via (bv, bi);
 * every thread leaves with the same answer. Requires __syncthreads() before the next use
 * of the shared scratch (callers are sync-heavy anyway; the round loop syncs per round). */
__device__ __forceinline__ bool r3d_keep_all(float, int64_t) { return true; }

/* topk_next's window, as a functor: candidates strictly after the previous pick in the
 * (v desc, id asc) order -- ref skips `v > pv` and `v == pv && i <= pi`. Namespace scope
 * because a kernel-local type is a fragile template argument for device codegen. */
struct R3dAfter {
    float pv; int64_t pi;
    __device__ bool operator()(float v, int64_t i) const {
        if (v > pv) return false;
        if (v == pv && i <= pi) return false;
        return true;
    }
};

template <typename Keep>
__device__ void r3d_block_argmax(const float* __restrict__ row, int64_t n, Keep keep,
                                 float* s_v, int64_t* s_i, float* bv_out, int64_t* bi_out) {
    float bv = 0.0f; int64_t bi = -1;
    for (int64_t i = (int64_t)threadIdx.x; i < n; i += (int64_t)kThreads) {
        const float v = row[i];
        if (keep(v, i) && r3d_beats(v, i, bv, bi)) { bv = v; bi = i; }
    }
    /* intra-wave reduce (wave32: full mask). */
    for (uint32_t off = 16; off; off >>= 1) {
        const float sv = __shfl_down_sync(0xFFFFFFFFull, bv, off);
        const int64_t si = __shfl_down_sync(0xFFFFFFFFull, bi, off);
        if (r3d_beats(sv, si, bv, bi)) { bv = sv; bi = si; }
    }
    const uint32_t lane = threadIdx.x & 31u, wid = threadIdx.x >> 5;
    if (lane == 0) { s_v[wid] = bv; s_i[wid] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float rv = s_v[0]; int64_t ri = s_i[0];
        for (uint32_t w = 1; w < kWarps; ++w)
            if (r3d_beats(s_v[w], s_i[w], rv, ri)) { rv = s_v[w]; ri = s_i[w]; }
        s_v[0] = rv; s_i[0] = ri;
    }
    __syncthreads();
    *bv_out = s_v[0]; *bi_out = s_i[0];
}

/* ------------------------------------------------------------------ sample_temp */
__global__ __launch_bounds__(kThreads) void temp_kernel(
    float* __restrict__ lg, int64_t pitch, int64_t n_vocab,
    const RadSampleParams* __restrict__ p) {
    const RadSampleParams& sp = p[blockIdx.x];
    const float t = sp.temp;
    if (!(t > 0.0f) || t == 1.0f) return;   /* ref: untouched rows; greedy is argmax */
    const float inv = 1.0f / t;
    float* __restrict__ row = lg + (int64_t)blockIdx.x * pitch;
    for (int64_t i = (int64_t)threadIdx.x; i < n_vocab; i += (int64_t)kThreads)
        row[i] = row[i] * inv;
}

/* ------------------------------------------------------------------ sample_mask */
__global__ __launch_bounds__(kThreads) void mask_kernel(
    float* __restrict__ lg, int64_t pitch, int64_t n_vocab,
    const RadSampleParams* __restrict__ p,
    const uint32_t* __restrict__ bm, int64_t bm_pitch, int64_t mask_rows) {
    const RadSampleParams& sp = p[blockIdx.x];
    if (!(sp.flags & RAD_SP_MASK)) return;
    const int64_t row = sp.mask_row;
    if (row < 0 || row >= mask_rows) return;
    const uint32_t* __restrict__ bmr = bm + row * bm_pitch;
    float* __restrict__ lr = lg + (int64_t)blockIdx.x * pitch;
    for (int64_t i = (int64_t)threadIdx.x; i < n_vocab; i += (int64_t)kThreads) {
        if (!((bmr[i >> 5] >> (i & 31)) & 1u)) lr[i] = -INFINITY;
    }
}

/* ------------------------------------------------------------------ sample_argmax */
__global__ __launch_bounds__(kThreads) void argmax_kernel(
    const float* __restrict__ lg, int64_t pitch, int64_t n_vocab, int32_t* __restrict__ tok) {
    __shared__ float s_v[kWarps]; __shared__ int64_t s_i[kWarps];
    float bv; int64_t bi;
    r3d_block_argmax(lg + (int64_t)blockIdx.x * pitch, n_vocab, r3d_keep_all,
                     s_v, s_i, &bv, &bi);
    if (threadIdx.x == 0) tok[blockIdx.x] = (int32_t)bi;
}

/* ------------------------------------------------------------------ sample_topk
 * `pairs`, when present, is the vocab-parallel plane the all-gather reads: each slot
 * holds (u32 global id, f32 logit bits); the hole id is 0xffffffff (the merge reads the
 * pair plane as int32, so that reads -1) and the hole value is the bit pattern of
 * -infinity. Global ids are the local column plus vocab_off, exactly as ref packs them. */
__global__ __launch_bounds__(kThreads) void topk_kernel(
    const float* __restrict__ lg, int64_t lg_pitch, int64_t n_vocab,
    const RadSampleParams* __restrict__ p,
    int32_t* __restrict__ cidx, float* __restrict__ cval,
    int64_t ci_pitch, int32_t n_cand,
    uint32_t* __restrict__ pairs, int64_t pairs_pitch, int32_t voff) {
    __shared__ float s_v[kWarps]; __shared__ int64_t s_i[kWarps];
    __shared__ float s_pv; __shared__ int64_t s_pi;

    const int64_t m = blockIdx.x;
    const RadSampleParams& sp = p[m];
    int64_t k = sp.top_k;
    if (k <= 0 || k > n_cand) k = n_cand;      /* ref: <= 0 takes the whole width */
    if (sp.flags & RAD_SP_GREEDY) k = 0;       /* a greedy row gets the empty set */

    const float* __restrict__ row = lg + m * lg_pitch;
    int32_t* __restrict__ ir = cidx + m * ci_pitch;
    float*   __restrict__ vr = cval + m * ci_pitch;
    uint32_t* __restrict__ pr = pairs ? pairs + m * pairs_pitch : nullptr;

    /* Holes first; the rounds overwrite the first k. */
    for (int32_t r = (int32_t)threadIdx.x; r < n_cand; r += (int64_t)kThreads) {
        ir[r] = -1; vr[r] = -INFINITY;
        if (pr) { pr[2 * r] = 0xffffffffu; pr[2 * r + 1] = 0xff800000u; }  /* bits(-inf) */
    }
    __syncthreads();
    if (threadIdx.x == 0) { s_pv = INFINITY; s_pi = -1; }
    __syncthreads();

    for (int64_t r = 0; r < k; ++r) {
        float bv; int64_t bi;
        r3d_block_argmax(row, n_vocab, R3dAfter{s_pv, s_pi}, s_v, s_i, &bv, &bi);
        if (bi < 0) break;                      /* exhausted: the rest are holes */
        if (threadIdx.x == 0) {
            ir[r] = (int32_t)bi; vr[r] = bv;
            if (pr) {
                uint32_t bits;
                __builtin_memcpy(&bits, &bv, 4);
                pr[2 * r] = (uint32_t)((int32_t)bi + voff);
                pr[2 * r + 1] = bits;
            }
            s_pv = bv; s_pi = bi;
        }
        __syncthreads();
    }
}

/* ------------------------------------------------------------------ the narrowing stages and pick
 * All three run one thread per row: their f32 accumulation order IS the spec (ref_common's
 * summation rule and libr4d's row note), the candidate width is a few hundred at most, and
 * M is a batch of sampled positions -- a few thousand sequential f32 adds per row is noise
 * next to a forward pass, and buying bit agreement with the oracle is the whole point. */

/* Max over live candidates (cand_idx >= 0), in-order f32 with ref's `>` (first max wins). */
__device__ __forceinline__ float narrow_max(const int32_t* __restrict__ ir,
                                            const float* __restrict__ vr, int32_t n_cand) {
    float mx = -INFINITY;
    for (int32_t i = 0; i < n_cand; ++i)
        if (ir[i] >= 0 && vr[i] > mx) mx = vr[i];
    return mx;
}
__device__ __forceinline__ void narrow_drop(int32_t* __restrict__ ir,
                                            float* __restrict__ vr, int32_t i) {
    ir[i] = -1; vr[i] = -INFINITY;
}

__global__ __launch_bounds__(kThreads) void narrow_kernel(
    int32_t* __restrict__ cidx, float* __restrict__ cval,
    const RadSampleParams* __restrict__ p,
    int64_t ci_pitch, int32_t n_cand, int stage) {
    extern __shared__ float smf[];   /* stage 2: q at [0], d at [n_cand] */
    if (threadIdx.x != 0) return;
    const int64_t m = blockIdx.x;
    const RadSampleParams& sp = p[m];
    if (sp.flags & RAD_SP_GREEDY) return;       /* narrowing leaves the greedy row alone */
    int32_t* __restrict__ ir = cidx + m * ci_pitch;
    float*   __restrict__ vr = cval + m * ci_pitch;

    const float mx = narrow_max(ir, vr, n_cand);
    if (mx == -INFINITY) return;

    if (stage == 1) {                           /* min-p: exp ratio vs the max, no norm */
        const float pp = sp.min_p;
        if (!(pp > 0.0f)) return;
        for (int32_t i = 0; i < n_cand; ++i) {
            if (ir[i] < 0) continue;
            const float v = vr[i];
            if (v == mx) continue;              /* the max always survives */
            if (r3d_exp(v - mx) < pp) narrow_drop(ir, vr, i);
        }
        return;
    }

    /* stage 0 (top-p) needs the normaliser: sum of exp(v - mx) over live, in order. */
    float sum = 0.0f;
    for (int32_t i = 0; i < n_cand; ++i)
        if (ir[i] >= 0) sum += r3d_exp(vr[i] - mx);
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    if (stage == 2) {               /* typical: a threshold on |surprisal - H|, ties kept */
        const float tp = sp.typical_p;
        if (!(tp < 1.0f) || !(tp > 0.0f)) return;
        float* q = smf;
        float* d = smf + n_cand;
        for (int32_t i = 0; i < n_cand; ++i) { q[i] = 0.0f; d[i] = 0.0f; }
        float H = 0.0f;
        for (int32_t i = 0; i < n_cand; ++i) {
            if (ir[i] < 0) continue;
            q[i] = r3d_exp(vr[i] - mx) * inv;
            if (q[i] > 0.0f) H -= q[i] * r3d_log(q[i]);
        }
        for (int32_t i = 0; i < n_cand; ++i)
            if (q[i] > 0.0f) d[i] = fabsf(-r3d_log(q[i]) - H);
        /* The cut is the SMALLEST distance whose at-or-below mass reaches p; ref scans the
         * distances that occur, in candidate order, keeping the minimum that qualifies --
         * an O(n^2) scan, and n_cand <= 4096 on one thread is the price of the tie rule. */
        float cut = 0.0f;
        bool have_cut = false;
        for (int32_t c = 0; c < n_cand; ++c) {
            if (!(q[c] > 0.0f)) continue;
            float mass = 0.0f;
            for (int32_t i = 0; i < n_cand; ++i)
                if (q[i] > 0.0f && d[i] <= d[c]) mass += q[i];
            if (mass >= tp && (!have_cut || d[c] < cut)) { cut = d[c]; have_cut = true; }
        }
        if (!have_cut) return;      /* rounding left the whole mass below p: nothing is cut */
        for (int32_t i = 0; i < n_cand; ++i)
            if (ir[i] >= 0 && (!(q[i] > 0.0f) || d[i] > cut)) narrow_drop(ir, vr, i);
        return;
    }

    if (stage == 3) {               /* xtc: drop above-threshold candidates but the last one */
        if (!(sp.xtc_probability > 0.0f) || !(sp.xtc_threshold > 0.0f)) return;
        if (r3d_uniform01(sp.seed ^ RAD_SAMPLE_KEY_XTC, sp.pos) >= sp.xtc_probability)
            return;                 /* the row's own stream decides whether the stage fires */
        int64_t last_above = -1, n_above = 0;
        for (int32_t i = 0; i < n_cand; ++i) {
            if (ir[i] < 0) continue;
            if (r3d_exp(vr[i] - mx) * inv >= sp.xtc_threshold) { last_above = i; ++n_above; }
        }
        if (n_above < 2) return;    /* needs two above to have anything to drop */
        for (int32_t i = 0; i < n_cand; ++i) {
            if (ir[i] < 0 || i == last_above) continue;
            if (r3d_exp(vr[i] - mx) * inv >= sp.xtc_threshold) narrow_drop(ir, vr, i);
        }
        return;
    }

    const float pp = sp.top_p;
    if (!(pp < 1.0f) || !(pp > 0.0f)) return;
    /* Keep the shortest prefix whose cumulative probability reaches p; the candidates
     * arrive descending from top-k and holes are skipped, never compacted. */
    float cum = 0.0f;
    bool done = false;
    for (int32_t i = 0; i < n_cand; ++i) {
        if (ir[i] < 0) continue;
        if (done) { narrow_drop(ir, vr, i); continue; }
        cum += r3d_exp(vr[i] - mx) * inv;
        if (cum >= pp) done = true;
    }
}

/* ------------------------------------------------------------------ sample_pick */
__global__ __launch_bounds__(kThreads) void pick_kernel(
    const int32_t* __restrict__ cidx, const float* __restrict__ cval,
    const RadSampleParams* __restrict__ p, int32_t* __restrict__ tok,
    int64_t ci_pitch, int32_t n_cand) {
    if (threadIdx.x != 0) return;
    const int64_t m = blockIdx.x;
    if (p[m].flags & RAD_SP_GREEDY) return;     /* the token is the argmax's already */
    const int32_t* __restrict__ ir = cidx + m * ci_pitch;
    const float*   __restrict__ vr = cval + m * ci_pitch;

    const float mx = narrow_max(ir, vr, n_cand);
    if (mx == -INFINITY) { tok[m] = -1; return; }

    float sum = 0.0f;
    for (int32_t i = 0; i < n_cand; ++i)
        if (ir[i] >= 0) sum += r3d_exp(vr[i] - mx);
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    const float u = r3d_uniform01(p[m].seed ^ RAD_SAMPLE_KEY_PICK, p[m].pos);

    /* Inverse CDF in candidate order. A draw that falls through (f32 sum < u < 1) takes
     * the last candidate ABOVE -infinity -- masked tail entries must never be it (ref). */
    float cum = 0.0f;
    int64_t chosen = -1, last = -1;
    for (int32_t i = 0; i < n_cand; ++i) {
        const int32_t id = ir[i];
        if (id < 0) continue;
        const float v = vr[i];
        if (v > -INFINITY) last = id;
        cum += r3d_exp(v - mx) * inv;
        if (u < cum) { chosen = id; break; }
    }
    tok[m] = (int32_t)(chosen >= 0 ? chosen : last);
}

/* ------------------------------------------------------------------ sample_penalties
 * One thread walks the row's history window; the work is per DISTINCT token in a window of
 * at most `hmax` ids, not per vocabulary column, so one block per row with thread 0 doing
 * the walk is the whole kernel. Ref's per-distinct-token arithmetic is line-mirrored:
 * repetition divides above zero and MULTIPLIES at or below (dividing a negative logit
 * would raise it), then the count times freq, then pres -- two separate subtractions, the
 * order ref writes them, not one fused expression. */
__global__ __launch_bounds__(kThreads) void penalties_kernel(
    float* __restrict__ lg, int64_t lg_pitch, int64_t n_vocab,
    const RadSampleParams* __restrict__ p,
    const int32_t* __restrict__ hist, int64_t hist_pitch,
    int32_t voff) {
    if (threadIdx.x != 0) return;
    const int64_t m = blockIdx.x;
    const RadSampleParams& sp = p[m];
    if (!(sp.flags & RAD_SP_PENALTY)) return;
    const float rep = sp.rep_penalty, freq = sp.freq_penalty, pres = sp.pres_penalty;
    if (rep == 1.0f && freq == 0.0f && pres == 0.0f) return;

    const int64_t hmax = hist_pitch;           /* entry checked numel/M == shape[1] */
    int64_t ho = 0, hl = 0;
    if (!hist_window(sp, sp.penalty_last_n, hmax, &ho, &hl)) return;

    const int32_t* __restrict__ hr = hist + m * hist_pitch + ho;
    float* __restrict__ lr = lg + m * lg_pitch;
    for (int64_t i = 0; i < hl; ++i) {
        const int64_t tok = hr[i];
        const int64_t col = tok - voff;
        if (tok < 0 || col < 0 || col >= n_vocab) continue;
        /* First occurrence only; the count carries the multiplicity. */
        bool first = true;
        int64_t count = 0;
        for (int64_t j = 0; j < hl; ++j) {
            if (hr[j] != tok) continue;
            if (j < i) { first = false; break; }
            ++count;
        }
        if (!first) continue;
        float v = lr[col];
        v = v > 0.0f ? v / rep : v * rep;
        v -= (float)count * freq;
        v -= pres;
        lr[col] = v;
    }
}

/* ------------------------------------------------------------------ sample_dry
 * The naive longest-suffix scan, mirrored from ref_sample_dry (NOT host_ref's Z-algorithm
 * -- the oracle is the plugin). Distinct continuation tokens and their longest repeat live
 * in dynamic LDS (8 bytes a window slot); the entry refuses windows past 6144. The penalty
 * is the repeated-multiplication form with ref's FLT_MAX/base stop, so every
 * implementation computes the same float. */
__global__ __launch_bounds__(kThreads) void dry_kernel(
    float* __restrict__ lg, int64_t lg_pitch, int64_t n_vocab,
    const RadSampleParams* __restrict__ p,
    const int32_t* __restrict__ hist, int64_t hist_pitch,
    const int32_t* __restrict__ brk, int64_t brk_pitch,
    int32_t voff) {
    extern __shared__ int32_t sm32[];          /* [hl] tokens then [hl] lengths */
    if (threadIdx.x != 0) return;
    const int64_t m = blockIdx.x;
    const RadSampleParams& sp = p[m];
    if (!(sp.flags & RAD_SP_DRY) || !(sp.dry_multiplier > 0.0f)) return;
    const int64_t allowed = sp.dry_allowed_length > 0 ? sp.dry_allowed_length : 2;

    int64_t ho = 0, hl = 0;
    if (!hist_window(sp, sp.dry_penalty_last_n, hist_pitch, &ho, &hl)) return;
    if (hl <= allowed) return;
    const int64_t cap = sp.dry_rep_limit > 0 && sp.dry_rep_limit < hl ? sp.dry_rep_limit : hl;
    if (cap < allowed) return;

    const int32_t* __restrict__ hr = hist + m * hist_pitch + ho;
    const int32_t* __restrict__ br = brk ? brk + m * brk_pitch + ho : nullptr;
    float* __restrict__ lr = lg + m * lg_pitch;
    int32_t* __restrict__ tl = sm32;
    int32_t* __restrict__ ll = sm32 + hl;
    int64_t nd = 0;

    for (int64_t j = 0; j + 1 < hl; ++j) {
        const int64_t lim = j + 1 < cap ? j + 1 : cap;
        int64_t n = 0;
        while (n < lim && hr[j - n] == hr[hl - 1 - n]) ++n;
        if (n < allowed || (br && br[j + 1] != 0)) continue;   /* exempt: a breaker ends it */
        const int32_t next = hr[j + 1];
        const int64_t col = (int64_t)next - voff;
        if (next < 0 || col < 0 || col >= n_vocab) continue;
        int64_t found = -1;
        for (int64_t t = 0; t < nd; ++t) if (tl[t] == next) { found = t; break; }
        if (found < 0) { tl[nd] = next; ll[nd] = (int32_t)n; ++nd; }
        else if (n > ll[found]) ll[found] = (int32_t)n;
    }

    const float base = sp.dry_base > 0.0f ? sp.dry_base : 1.75f;
    const float lim = 3.402823466e+38f / base;                 /* FLT_MAX / base */
    for (int64_t t = 0; t < nd; ++t) {
        float pen = sp.dry_multiplier;
        if (base != 1.0f)
            for (int64_t e = 0; e < (int64_t)ll[t] - allowed; ++e) {
                if (pen > lim) break;
                pen *= base;
            }
        float* off = lr + ((int64_t)tl[t] - voff);
        *off = *off - pen;
    }
}

/* ------------------------------------------------------------------ sample_logprobs
 * The raw distribution's summary: max, sum of exp(l - mx) in DOUBLE, and the k largest as
 * (id, logit) pairs under the one total order -- k rounds of block argmax after the last
 * pick, exactly the topk_kernel's selection. Rows without the flag are not written.
 * The double sum is a tree, not ref's ascending walk: regrouping a double accumulation
 * moves the f32 narrowing by at most an ulp, which kbench reads as 1e-07 rel_l2, nowhere
 * near a decision boundary (unlike the pick, this sum feeds no threshold). */
__global__ __launch_bounds__(kThreads) void logprobs_kernel(
    const float* __restrict__ lg, int64_t pitch, int64_t n_vocab,
    const RadSampleParams* __restrict__ p,
    float* __restrict__ st, int64_t st_pitch, int32_t k, int32_t voff) {
    __shared__ float s_v[kWarps]; __shared__ int64_t s_i[kWarps];
    __shared__ float s_pv; __shared__ int64_t s_pi;
    __shared__ double s_d[kThreads];

    const int64_t m = blockIdx.x;
    if (!(p[m].flags & RAD_SP_LOGPROBS)) return;
    const float* __restrict__ row = lg + m * pitch;
    float* __restrict__ sr = st + m * st_pitch;

    float bv; int64_t bi;
    r3d_block_argmax(row, n_vocab, r3d_keep_all, s_v, s_i, &bv, &bi);
    const float mx = bv;

    double part = 0.0;
    for (int64_t i = (int64_t)threadIdx.x; i < n_vocab; i += (int64_t)kThreads)
        part += exp((double)row[i] - (double)mx);
    s_d[threadIdx.x] = part;
    __syncthreads();
    for (uint32_t off = kThreads / 2; off; off >>= 1) {
        if (threadIdx.x < off) s_d[threadIdx.x] += s_d[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) { sr[0] = mx; sr[1] = (float)s_d[0]; }

    if (threadIdx.x == 0) { s_pv = INFINITY; s_pi = -1; }
    __syncthreads();
    for (int32_t j = 0; j < k; ++j) {
        float vv; int64_t vi;
        r3d_block_argmax(row, n_vocab, R3dAfter{s_pv, s_pi}, s_v, s_i, &vv, &vi);
        if (vi < 0) {                      /* vocabulary exhausted: the rest are holes */
            for (int32_t jj = j + (int32_t)threadIdx.x; jj < k; jj += (int64_t)kThreads) {
                sr[2 + 2 * jj] = -1.0f;
                sr[3 + 2 * jj] = -INFINITY;
            }
            break;
        }
        if (threadIdx.x == 0) {
            sr[2 + 2 * j] = (float)((int32_t)vi + voff);   /* ids exact below 2^24 */
            sr[3 + 2 * j] = vv;
            s_pv = vv; s_pi = vi;
        }
        __syncthreads();
    }
}

/* ------------------------------------------------------------------ host-side helpers */
/* params: the byte blob the builder declares as [rows, 24] u32 (ref's check, same rule:
 * a 32-bit dtype or the row reads wrong, and the byte count must cover M rows). */
static inline bool param_rows_ok(const RadTensor* t, int64_t M) {
    if (!t || !t->data) return false;
    if (t->dtype != RAD_U32 && t->dtype != RAD_I32 && t->dtype != RAD_F32) return false;
    return rad_tensor_numel(t) * 4 >= M * (int64_t)sizeof(RadSampleParams);
}
/* rank-2 row-major-with-pitch reader for logits / candidate planes. */
static inline int row2(const RadTensor* t, int64_t cols, int64_t* pitch) {
    if (!t || !t->data) return -1;
    if (t->rank != 2) return -2;
    if (t->shape[1] != cols) return -3;
    if (t->stride[1] != 1) return -4;
    *pitch = t->stride[0];
    return 0;
}

} /* anonymous namespace */

/* ------------------------------------------------------------------ entries */
extern "C" int r3d_rad_sample_temp(const RadArgs* a, RadStream s) {
    const RadTensor* lg = rad_arg_in(a, 0);
    const RadTensor* pr = rad_arg_in(a, 1);
    if (!lg || !pr) return RAD_E_INVAL;
    if (lg->dtype != RAD_F32) return RAD_E_DTYPE;
    long long nv = 0;
    if (!rad_args_geti(a, "n_vocab", &nv) || nv <= 0) nv = lg->rank >= 2 ? lg->shape[1] : 0;
    if (nv <= 0) return RAD_E_SHAPE;
    const int64_t M = rad_tensor_numel(lg) / nv;
    if (M <= 0 || M > 0xffff) return RAD_E_SHAPE;
    if (!param_rows_ok(pr, M)) return RAD_E_SHAPE;
    int64_t pitch; if (row2(lg, nv, &pitch)) return RAD_E_STRIDE;
    hipLaunchKernelGGL(temp_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                       (hipStream_t)(uintptr_t)s, (float*)lg->data, pitch, nv,
                       (const RadSampleParams*)pr->data);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r3d_rad_sample_mask(const RadArgs* a, RadStream s) {
    const RadTensor* lg = rad_arg_in(a, 0);
    const RadTensor* pr = rad_arg_in(a, 1);
    const RadTensor* bm = rad_arg_in(a, 2);
    if (!lg || !pr || !bm) return RAD_E_INVAL;
    if (lg->dtype != RAD_F32) return RAD_E_DTYPE;
    if (bm->dtype != RAD_U32 && bm->dtype != RAD_I32) return RAD_E_DTYPE;
    long long nv = 0;
    if (!rad_args_geti(a, "n_vocab", &nv) || nv <= 0) nv = lg->rank >= 2 ? lg->shape[1] : 0;
    if (nv <= 0) return RAD_E_SHAPE;
    const int64_t M = rad_tensor_numel(lg) / nv;
    if (M <= 0 || M > 0xffff) return RAD_E_SHAPE;
    if (!param_rows_ok(pr, M)) return RAD_E_SHAPE;
    int64_t pitch; if (row2(lg, nv, &pitch)) return RAD_E_STRIDE;
    const int64_t words = (nv + 31) / 32;
    int64_t bm_pitch;
    if (bm->rank != 2 || bm->shape[1] != words || bm->stride[1] != 1) return RAD_E_STRIDE;
    bm_pitch = bm->stride[0];
    const int64_t mask_rows = rad_tensor_numel(bm) / words;
    if (mask_rows <= 0) return RAD_E_SHAPE;
    hipLaunchKernelGGL(mask_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                       (hipStream_t)(uintptr_t)s, (float*)lg->data, pitch, nv,
                       (const RadSampleParams*)pr->data,
                       (const uint32_t*)bm->data, bm_pitch, mask_rows);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r3d_rad_sample_argmax(const RadArgs* a, RadStream s) {
    const RadTensor* lg   = rad_arg_in(a, 0);
    const RadTensor* tok  = rad_arg_in(a, 2);
    if (!lg || !tok) return RAD_E_INVAL;
    if (lg->dtype != RAD_F32 || tok->dtype != RAD_I32) return RAD_E_DTYPE;
    long long nv = 0;
    if (!rad_args_geti(a, "n_vocab", &nv) || nv <= 0) nv = lg->rank >= 2 ? lg->shape[1] : 0;
    if (nv <= 0) return RAD_E_SHAPE;
    const int64_t M = rad_tensor_numel(lg) / nv;
    if (M <= 0 || M > 0xffff) return RAD_E_SHAPE;
    if (rad_tensor_numel(tok) < M || tok->rank != 1 || tok->stride[0] != 1) return RAD_E_SHAPE;
    int64_t pitch; if (row2(lg, nv, &pitch)) return RAD_E_STRIDE;
    hipLaunchKernelGGL(argmax_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                       (hipStream_t)(uintptr_t)s, (const float*)lg->data, pitch, nv,
                       (int32_t*)tok->data);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r3d_rad_sample_topk(const RadArgs* a, RadStream s) {
    const RadTensor* lg  = rad_arg_in(a, 0);
    const RadTensor* pr  = rad_arg_in(a, 1);
    const RadTensor* ci  = rad_arg_in(a, 2);
    const RadTensor* cv  = rad_arg_in(a, 3);
    if (!lg || !pr || !ci || !cv) return RAD_E_INVAL;
    /* The vocab-parallel pairs plane is written when present (the 27B-class fixtures were
     * recorded under a sharded lm_head); its width is 2 * n_cand words. */
    const RadTensor* pp = (a->n_t > 4 && a->t[4].data) ? &a->t[4] : nullptr;
    if (lg->dtype != RAD_F32 || ci->dtype != RAD_I32 || cv->dtype != RAD_F32) return RAD_E_DTYPE;
    if (pp && pp->dtype != RAD_U32 && pp->dtype != RAD_I32) return RAD_E_DTYPE;
    long long nv = 0;
    if (!rad_args_geti(a, "n_vocab", &nv) || nv <= 0) nv = lg->rank >= 2 ? lg->shape[1] : 0;
    if (nv <= 0) return RAD_E_SHAPE;
    const int64_t M = rad_tensor_numel(lg) / nv;
    if (M <= 0 || M > 0xffff) return RAD_E_SHAPE;
    if (!param_rows_ok(pr, M)) return RAD_E_SHAPE;
    /* The candidate width is the buffer's, not the request's (ref's rule). */
    if (ci->rank != 2) return RAD_E_SHAPE;
    const int32_t n_cand = (int32_t)ci->shape[1];
    if (n_cand <= 0) return RAD_E_SHAPE;
    if (rad_tensor_numel(cv) < M * n_cand) return RAD_E_SHAPE;
    int64_t lg_p, ci_p, cv_p;
    if (row2(lg, nv, &lg_p)) return RAD_E_STRIDE;
    if (row2(ci, n_cand, &ci_p)) return RAD_E_STRIDE;
    if (row2(cv, n_cand, &cv_p) || cv_p != ci_p) return RAD_E_STRIDE;
    int64_t pp_p = 0;
    if (pp) {
        if (row2(pp, 2 * (long long)n_cand, &pp_p)) return RAD_E_STRIDE;
        if (rad_tensor_numel(pp) < M * 2 * (int64_t)n_cand) return RAD_E_SHAPE;
    }
    long long voff = 0;
    rad_args_geti(a, "vocab_off", &voff);
    hipLaunchKernelGGL(topk_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                       (hipStream_t)(uintptr_t)s, (const float*)lg->data, lg_p, nv,
                       (const RadSampleParams*)pr->data,
                       (int32_t*)ci->data, (float*)cv->data, ci_p, n_cand,
                       pp ? (uint32_t*)pp->data : nullptr, pp_p, (int32_t)voff);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* narrow: stage 0 = top-p, 1 = min-p, 2 = typical, 3 = xtc (same operand shape, all in
 * place). Typical stages q/d for the row in LDS: two f32 arrays of n_cand. */
static int sample_narrow(const RadArgs* a, RadStream s, int stage) {
    const RadTensor* ci = rad_arg_in(a, 0);
    const RadTensor* cv = rad_arg_in(a, 1);
    const RadTensor* pr = rad_arg_in(a, 2);
    if (!ci || !cv || !pr) return RAD_E_INVAL;
    if (ci->dtype != RAD_I32 || cv->dtype != RAD_F32) return RAD_E_DTYPE;
    long long nc = 0;
    if (!rad_args_geti(a, "n_cand", &nc) || nc <= 0) nc = ci->rank >= 2 ? ci->shape[1] : 0;
    if (nc <= 0 || nc > 4096) return RAD_E_SHAPE;
    const int32_t n_cand = (int32_t)nc;
    const int64_t M = rad_tensor_numel(ci) / nc;
    if (M <= 0 || M > 0xffff) return RAD_E_SHAPE;
    if (rad_tensor_numel(cv) < M * nc) return RAD_E_SHAPE;
    if (!param_rows_ok(pr, M)) return RAD_E_SHAPE;
    int64_t ci_p, cv_p;
    if (row2(ci, n_cand, &ci_p)) return RAD_E_STRIDE;
    if (row2(cv, n_cand, &cv_p) || cv_p != ci_p) return RAD_E_STRIDE;
    const unsigned smem = stage == 2 ? (unsigned)(2 * n_cand * (int64_t)sizeof(float)) : 0u;
    hipLaunchKernelGGL(narrow_kernel, dim3((uint32_t)M), dim3(kThreads), smem,
                       (hipStream_t)(uintptr_t)s, (int32_t*)ci->data, (float*)cv->data,
                       (const RadSampleParams*)pr->data, ci_p, n_cand, stage);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
extern "C" int r3d_rad_sample_topp(const RadArgs* a, RadStream s) { return sample_narrow(a, s, 0); }
extern "C" int r3d_rad_sample_minp(const RadArgs* a, RadStream s) { return sample_narrow(a, s, 1); }
extern "C" int r3d_rad_sample_typical(const RadArgs* a, RadStream s) { return sample_narrow(a, s, 2); }
extern "C" int r3d_rad_sample_xtc(const RadArgs* a, RadStream s) { return sample_narrow(a, s, 3); }

extern "C" int r3d_rad_sample_pick(const RadArgs* a, RadStream s) {
    const RadTensor* ci  = rad_arg_in(a, 0);
    const RadTensor* cv  = rad_arg_in(a, 1);
    const RadTensor* pr  = rad_arg_in(a, 2);
    const RadTensor* tok = rad_arg_in(a, 3);
    if (!ci || !cv || !pr || !tok) return RAD_E_INVAL;
    if (ci->dtype != RAD_I32 || cv->dtype != RAD_F32 || tok->dtype != RAD_I32) return RAD_E_DTYPE;
    long long nc = 0;
    if (!rad_args_geti(a, "n_cand", &nc) || nc <= 0) nc = ci->rank >= 2 ? ci->shape[1] : 0;
    if (nc <= 0 || nc > 4096) return RAD_E_SHAPE;
    const int32_t n_cand = (int32_t)nc;
    const int64_t M = rad_tensor_numel(ci) / nc;
    if (M <= 0 || M > 0xffff) return RAD_E_SHAPE;
    if (rad_tensor_numel(cv) < M * nc || rad_tensor_numel(tok) < M) return RAD_E_SHAPE;
    if (tok->rank != 1 || tok->stride[0] != 1) return RAD_E_SHAPE;
    if (!param_rows_ok(pr, M)) return RAD_E_SHAPE;
    int64_t ci_p, cv_p;
    if (row2(ci, n_cand, &ci_p)) return RAD_E_STRIDE;
    if (row2(cv, n_cand, &cv_p) || cv_p != ci_p) return RAD_E_STRIDE;
    hipLaunchKernelGGL(pick_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                       (hipStream_t)(uintptr_t)s, (const int32_t*)ci->data,
                       (const float*)cv->data, (const RadSampleParams*)pr->data,
                       (int32_t*)tok->data, ci_p, n_cand);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* ---- the history and summary stages. The history plane is [M, hmax] i32 contiguous:
 * ref reads any ld_int width, the engine's idx planes are i32, and a padded row pitch
 * would make ref's own numel/M hmax ambiguous -- all refused, not guessed. ---- */
static int hist_plane(const RadTensor* t, int64_t M, int64_t* pitch) {
    if (!t || !t->data || t->dtype != RAD_I32) return -1;
    if (t->rank != 2 || t->stride[1] != 1 || t->stride[0] != t->shape[1]) return -2;
    if (t->shape[1] <= 0 || rad_tensor_numel(t) < M * t->shape[1]) return -3;
    *pitch = t->stride[0];
    return 0;
}

extern "C" int r3d_rad_sample_penalties(const RadArgs* a, RadStream s) {
    const RadTensor* lg = rad_arg_in(a, 0);
    const RadTensor* pr = rad_arg_in(a, 1);
    const RadTensor* hi = rad_arg_in(a, 2);
    if (!lg || !pr || !hi) return RAD_E_INVAL;
    if (lg->dtype != RAD_F32) return RAD_E_DTYPE;
    long long nv = 0;
    if (!rad_args_geti(a, "n_vocab", &nv) || nv <= 0) nv = lg->rank >= 2 ? lg->shape[1] : 0;
    if (nv <= 0) return RAD_E_SHAPE;
    const int64_t M = rad_tensor_numel(lg) / nv;
    if (M <= 0 || M > 0xffff) return RAD_E_SHAPE;
    if (!param_rows_ok(pr, M)) return RAD_E_SHAPE;
    int64_t lg_p; if (row2(lg, nv, &lg_p)) return RAD_E_STRIDE;
    int64_t hi_p; if (hist_plane(hi, M, &hi_p)) return RAD_E_SHAPE;
    long long voff = 0; rad_args_geti(a, "vocab_off", &voff);
    hipLaunchKernelGGL(penalties_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                       (hipStream_t)(uintptr_t)s, (float*)lg->data, lg_p, nv,
                       (const RadSampleParams*)pr->data, (const int32_t*)hi->data, hi_p,
                       (int32_t)voff);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r3d_rad_sample_dry(const RadArgs* a, RadStream s) {
    const RadTensor* lg = rad_arg_in(a, 0);
    const RadTensor* pr = rad_arg_in(a, 1);
    const RadTensor* hi = rad_arg_in(a, 2);
    const RadTensor* bk = (a->n_t > 3 && a->t[3].data) ? &a->t[3] : nullptr;
    if (!lg || !pr || !hi) return RAD_E_INVAL;
    if (lg->dtype != RAD_F32) return RAD_E_DTYPE;
    long long nv = 0;
    if (!rad_args_geti(a, "n_vocab", &nv) || nv <= 0) nv = lg->rank >= 2 ? lg->shape[1] : 0;
    if (nv <= 0) return RAD_E_SHAPE;
    const int64_t M = rad_tensor_numel(lg) / nv;
    if (M <= 0 || M > 0xffff) return RAD_E_SHAPE;
    if (!param_rows_ok(pr, M)) return RAD_E_SHAPE;
    int64_t lg_p; if (row2(lg, nv, &lg_p)) return RAD_E_STRIDE;
    int64_t hi_p; if (hist_plane(hi, M, &hi_p)) return RAD_E_SHAPE;
    /* The distinct-token map is two int32 arrays of the window, in LDS. */
    if (hi_p > 6144) return RAD_E_SHAPE;
    int64_t bk_p = 0;
    if (bk && hist_plane(bk, M, &bk_p)) return RAD_E_SHAPE;
    long long voff = 0; rad_args_geti(a, "vocab_off", &voff);
    hipLaunchKernelGGL(dry_kernel, dim3((uint32_t)M), dim3(kThreads),
                       (unsigned)(2 * hi_p * (int64_t)sizeof(int32_t)),
                       (hipStream_t)(uintptr_t)s, (float*)lg->data, lg_p, nv,
                       (const RadSampleParams*)pr->data, (const int32_t*)hi->data, hi_p,
                       bk ? (const int32_t*)bk->data : nullptr, bk_p, (int32_t)voff);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r3d_rad_sample_logprobs(const RadArgs* a, RadStream s) {
    const RadTensor* lg = rad_arg_in(a, 0);
    const RadTensor* pr = rad_arg_in(a, 1);
    const RadTensor* st = rad_arg_in(a, 2);
    if (!lg || !pr || !st) return RAD_E_INVAL;
    if (lg->dtype != RAD_F32 || st->dtype != RAD_F32) return RAD_E_DTYPE;
    long long nv = 0;
    if (!rad_args_geti(a, "n_vocab", &nv) || nv <= 0) nv = lg->rank >= 2 ? lg->shape[1] : 0;
    if (nv <= 0) return RAD_E_SHAPE;
    long long k = 0;
    if (!rad_args_geti(a, "k", &k) || k <= 0 || k > 0xffff) return RAD_E_SHAPE;
    const int64_t M = rad_tensor_numel(lg) / nv;
    if (M <= 0 || M > 0xffff) return RAD_E_SHAPE;
    if (!param_rows_ok(pr, M)) return RAD_E_SHAPE;
    const int64_t w = 2 + 2 * k;
    int64_t lg_p, st_p;
    if (row2(lg, nv, &lg_p)) return RAD_E_STRIDE;
    if (row2(st, w, &st_p)) return RAD_E_STRIDE;
    if (rad_tensor_numel(st) < M * w) return RAD_E_SHAPE;
    long long voff = 0; rad_args_geti(a, "vocab_off", &voff);
    hipLaunchKernelGGL(logprobs_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                       (hipStream_t)(uintptr_t)s, (const float*)lg->data, lg_p, nv,
                       (const RadSampleParams*)pr->data, (float*)st->data, st_p,
                       (int32_t)k, (int32_t)voff);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
