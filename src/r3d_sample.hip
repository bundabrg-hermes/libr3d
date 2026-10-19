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
 * This file covers the classic chain the core issues for an ordinary sampled request:
 * sample_temp, sample_mask, sample_argmax, sample_topk, sample_topp, sample_minp,
 * sample_pick. sample_penalties / sample_dry / sample_typical / sample_xtc /
 * sample_logprobs (flag- and option-gated, so absent from default requests) stay on the
 * fallback until a later batch; the row table simply has no rows for them.
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

/* ref's transcendental rule: evaluate in double, round once. */
__device__ __forceinline__ float r3d_exp(float x) { return (float)exp((double)x); }

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

/* narrow: stage 0 = top-p, stage 1 = min-p (same operand shape, both in place). */
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
    hipLaunchKernelGGL(narrow_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                       (hipStream_t)(uintptr_t)s, (int32_t*)ci->data, (float*)cv->data,
                       (const RadSampleParams*)pr->data, ci_p, n_cand, stage);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
extern "C" int r3d_rad_sample_topp(const RadArgs* a, RadStream s) { return sample_narrow(a, s, 0); }
extern "C" int r3d_rad_sample_minp(const RadArgs* a, RadStream s) { return sample_narrow(a, s, 1); }

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
