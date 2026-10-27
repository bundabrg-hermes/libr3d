/* libr3d: split-KV paged-attention DECODE for RDNA3 (gfx1100/gfx1101).
 *
 * Second kernel row for op attn_paged, resolved instead of attn_paged_bf16 in the decode
 * band (window 0, q_len <= 8 -- decode and speculative verify alike). The measured why:
 * deep-context decode on the one-block-per-(head, query tile) WMMA tile runs 24 blocks
 * over 96 CUs and collapses to 0.5-15 GB/s -- 3.1 tok/s at 8k ctx (COVERAGE.md,
 * 2026-10-20). Split-KV turns one long KV walk into `splits` parallel slices: each slice
 * block runs THE SAME two-pass tile body (r3d_attn_wmma_body.inc, SPLITK=1) over its
 * share of the tiles and stores un-normalized partials (O, m, l) to the engine scratch
 * arena; a second, tiny merge kernel rescales the partials to one output row, per the
 * flash-decode pattern. Parallelism scales as q_heads * n_seq * splits instead of
 * q_heads * n_seq, and the tile keeps its LDS residency -- the fix the measurement
 * named, not a guess.
 *
 * Arithmetic identity: with one live split the merge degenerates to exactly the direct
 * out store of the SPLITK=0 tile (w = exp2(m-m) = 1.0 exactly, den = l, num = O, and the
 * divide-by-reciprocal-then-descale form is copied line for line), so a splits=1 launch
 * IS the old path -- which makes it the A/B arm, not a re-implementation of it. The
 * R3D_NO_SPLITK=1 kill switch forces exactly that.
 *
 * Partial layout in RadArgs.scratch (floats; the arena hook is
 * r3d_rad_attn_decode_scratch, sized at declare from n_head/max_seqs/max_ctx -- the two
 * parameters the attn_paged schema carries for precisely this purpose, per libr4d):
 *   O   so[idx*D + d]      un-normalized output column
 *   m   sm[idx]            the slice's row max (exp2 domain, as the tile tracks it)
 *   l   sl[idx]            its exp-sum
 *   idx = (tok*q_heads + h)*splits + sp, O plane first, then m, then l.
 * A slice with no tiles writes m=-inf, l=0 and no O bytes; the merge skips the O VALUE
 * on m==-inf (the load itself is speculative under Q46's kMergeU batching -- dead
 * register, never accumulated), so the inert splits cost no arithmetic and no partial
 * bytes on the write side.
 *
 * Shape law (host, one place): splits = min(wave / (q_heads*n_seq), cap, tiles_kv,
 * 64) with (wave, cap) = (128, tiles_kv) for the GQA-fold arm and (192, tiles_kv/2)
 * for the plain arm; an explicit `splits` (the row's tuner axis, or the engine
 * passing a tuned value) wins outright. Q46 (window 45): the closed form is now
 * R3D_SPLIT_POLICY=0; arm 1 replaces the (wave) target with the FA2 wave-efficiency
 * rule (short-circuit at 0.8*CUs blocks, else smallest s within 0.85 of best
 * wave-efficiency, same caps incl. the 64 that preserves the arena law) -- see
 * decode_splits below for the full rule and its precision class. The SAME
 * closed form sizes the arena at declare (swept over every seq count up to max_seqs,
 * because splits*seqs is not monotone in seqs). The fold-arm constants are the
 * 2026-10-24 retune: device-asserted paired sweeps (M 1..8, ctx 512..8192, 32Q/8KV
 * and 32Q/4KV, L2-cold, cat ~/scratch/splitsretune) put the fold optimum at ~128
 * blocks, not the plain arm's inherited two-wave 192 -- the slice walk is
 * latency-bound, so the extra slices buy wave-quantization tails instead of stream:
 * 16 beat the old 24 by 1.32-1.44x at every ctx>=2048 for 8KV heads, 32 beat the
 * old 48 by 1.40-1.43x at 8192 for 4KV heads, and the tiles_kv/2 cap had pinned
 * ctx-512 1.16x under its measured best (single-tile slices win there). The plain
 * arm never runs the shapes the sweep covered (gqa==1), so it keeps its constants.
 *
 * Fallbacks, all inside the row (a resolved row must serve, so nothing here declines
 * the op -- only the arithmetic path changes): foreign chip or R3D_NO_WMMA -> the scalar
 * reference port; splits resolving to 1, or an arena smaller than the law needs -> the
 * SPLITK=0 tile; both of those never launch the split kernel, so an untouched partial
 * buffer is not a correctness question. Same scalars every way: one r3d_attn_fill call.
 *
 * head_dim 256 (issue #11 milestone 3, behind R3D_HD256 like the prefill tile): the
 * fitted 256 body at SPLITK=1 joins as the dynamic-LDS slice arm (attribute raised per
 * instantiation, r3d_attn256_smem(PAD) bytes passed; PAD defaults to 1, same bench law
 * as the tile). bf16 KV and the v1 body only; splits=1 or a short arena falls back to
 * the m2 tile through the same tile_or_v1 path. The arena hook claims the 256 partials
 * only while the lever is set.
 */

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rad_abi.h"
#include "rad_plugin.h"
#include "r3d_attn.h"

/* r3d_attn_paged_wmma.hip: the single R3D_HD256 law (q8x value rule, accept
 * 2026-10-27) -- shared by the prefill launcher, this file's launcher AND the
 * decode arena hook so the three can never disagree about the lever state. */
extern "C" int r3d_hd256_on(void);

namespace {

/* dtype-widening helpers: bit-identical to the v1 kernel's and the prefill tile's (same
 * contracts, see r3d_attn_paged.hip); duplicated rather than shared to keep the TUs
 * independent -- these eleven lines are the most-reviewed eleven in the project. */
__device__ __forceinline__ float bf16_f32_w(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}
__device__ __forceinline__ uint16_t f32_bf16_w(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float fp8e4m3_w(unsigned b) {
    const unsigned e = (b >> 3) & 15u, m = b & 7u;
    float v = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0.001953125f;
    if (e == 15u && m == 7u) v = __uint_as_float(0x7fc00000u);
    return __uint_as_float(__float_as_uint(v) | ((b & 0x80u) << 24));
}
/* E4M3 -> bf16 bits: every code is a dyadic rational bf16 holds whole (4 significant
 * bits, exponent range -9..+8), so the narrowing is a truncation of an exact value. */
__device__ __forceinline__ uint16_t fp8_to_bf16_w(unsigned b) {
    return (uint16_t)(__float_as_uint(fp8e4m3_w(b)) >> 16);
}
__device__ __forceinline__ float ldt_w(const RadTensor* t, int64_t off) {
    switch (t->dtype) {
        case RAD_F32:  return ((const float*)t->data)[off];
        case RAD_BF16: return bf16_f32_w(((const uint16_t*)t->data)[off]);
        default:       return 0.0f;
    }
}
__device__ __forceinline__ float ldi_w(uint32_t dt, const void* p, int64_t i) {
    switch (dt) {
        case RAD_I32: return (int64_t)((const int32_t*)p)[i];
        case RAD_U32: return (int64_t)((const uint32_t*)p)[i];
        case RAD_I64: return ((const int64_t*)p)[i];
        default:      return 0;
    }
}

constexpr int kBM = 64, kBN = 32;           /* query rows x keys per step */
constexpr int kThreads = 128;               /* v1 body: 4 wave32s, 16 query rows each */
constexpr int kT = 256;                     /* v2 body (body2.inc): 8 wave32s */
constexpr float kLog2e = 1.44269504088896340736f;

} /* anonymous namespace */

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)

#include <rocwmma/rocwmma.hpp>

namespace {

namespace roCW = rocwmma;
using bf16 = roCW::bfloat16_t;

#include "r3d_attn_wmma_body.inc"    /* attn_paged_wmma_kernel<D, KVDT, SPLITK>  -- v1 schedule */
#include "r3d_attn_wmma_body2.inc"   /* attn_paged_wmma_kernel2<D, KVDT, SPLITK> -- v2 schedule */

} /* anonymous namespace */

#else  /* generic + non-RDNA3: the split kernel exists as a no-op so the symbol links;
        * the launcher's gcnArchName gate never dispatches it on such a chip. */

namespace {
template <int D, int KVDT, int SPLITK, int DBUF = 0, int FOLD = 0, int PAD = 0, int F8ST = 0, int SB = 0, int FS = 0>
__global__ __launch_bounds__(kThreads) void attn_paged_wmma_kernel(R3dAttnParams) {}
template <int D, int KVDT, int SPLITK>
__global__ __launch_bounds__(kT) void attn_paged_wmma_kernel2(R3dAttnParams) {}
} /* anonymous namespace */

#endif

namespace {

/* ---- merge: one output row per block, one thread per head_dim column ----
 * out = vds * sum_sp w_sp * O_sp / sum_sp w_sp * l_sp,  w_sp = exp2(m_sp - max m),
 * inert splits (m=-inf) skipped with no O VALU traffic. m/l addresses are warp-uniform,
 * so the skip predicate cannot diverge a wave. With one live split this is the tile's
 * own (O * (1/l)) * vds, to the rounding.
 *
 * Q46 (window 45) re-verification of the "bound the walk at n_act" half of the row,
 * on-card law reading: the shipped slice cut is the FLOOR cut (r3d_attn_wmma_body.inc
 * :445-456: a = t0 + nt*sp/spl, b = t0 + nt*(sp+1)/spl), so the LIVE splits of a row
 * are NOT a prefix of [0, splits). Worked counterexample to the donor's prefix bound
 * (nt=3 visible tiles, spl=8): cell edges floor(3k/8) = 0,0,0,1,1,2,2,2,3 -> live =
 * {2,4,6}: split 0 itself is inert and the last split spl-1 is ALWAYS live (its cell
 * is ceil(nt/spl) >= 1 tiles). "Splits beyond n_act are inert" is false for this cut
 * law, so a prefix walk bound would silently drop live splits -- the bound is NOT
 * shipped; walking to `splits` stays exact. What IS shipped from the donor is the
 * kMergeU=16 batching with the partial loads hoisted AHEAD of the skip-select (their
 * recorded fix: loads must not sink behind the branch): each batch loads m, l and O
 * for all 16 slots unconditionally into registers, and only the ACCUMULATE selects.
 * Bit-identical by construction: the accumulate set/order is unchanged (ascending sp,
 * live splits only -- inert still contributes nothing, not even a +0.0f add), so a
 * garbage O row behind an inert m=-inf is loaded into a dead register and never
 * touches the arithmetic. The only behavioural delta is arena READ traffic for inert
 * rows (they now cost the O row in loads, not in VALU), which exists only at
 * ctx < splits*kBN -- a few KB of scratch noise against the KV stream. */
constexpr int kMergeU = 16;    /* donor Strata qsa merge_v2 batch depth */

template <int D>
__global__ __launch_bounds__(D) void attn_splitk_merge_kernel(R3dAttnParams P, int splits) {
    const int h = blockIdx.x;
    const int64_t tok = blockIdx.y;
    const int d = threadIdx.x;

    /* the seq owning this query row: the decode band has few rows, so the walk v1 uses
     * is the cheap exact thing (no binary search machinery for <= 2048 rows). */
    int s = 0;
    if (P.cus) {
        for (; s < (int)P.n_seq; ++s)
            if (tok < (int64_t)ldi_w(P.cus_dt, P.cus, s + 1)) break;
    } else {
        s = (int)(tok / P.q_len_uniform);
    }

    const int64_t kh = h / P.gqa;
    const float vds = P.vdes ? ldt_w(P.vdes, s * P.kv_heads + kh) : 1.0f;
    const float* pso = reinterpret_cast<const float*>(P.pf);
    const int64_t pn = P.rows * P.q_heads;
    const float* psm = pso + pn * (int64_t)splits * D;
    const float* psl = psm + pn * (int64_t)splits;
    const int64_t base = (tok * P.q_heads + h) * splits;

    const int64_t ob = tok * P.o_rs + (int64_t)h * P.o_hs + (int64_t)d * P.o_ds;
    float mx = -INFINITY;
    for (int sp0 = 0; sp0 < splits; sp0 += kMergeU) {
        float mm[kMergeU];
        #pragma unroll
        for (int k = 0; k < kMergeU; ++k)          /* loads first: no select between */
            if (sp0 + k < splits) mm[k] = psm[base + sp0 + k];
        #pragma unroll
        for (int k = 0; k < kMergeU; ++k)
            if (sp0 + k < splits && mm[k] > mx) mx = mm[k];
    }
    if (mx == -INFINITY) {                  /* every split inert: the zero row, as v1 */
        if (P.out_f32) ((float*)P.out)[ob] = 0.0f;
        else           ((uint16_t*)P.out)[ob] = 0;
        return;
    }
    float num = 0.0f, den = 0.0f;
    for (int sp0 = 0; sp0 < splits; sp0 += kMergeU) {
        float mm[kMergeU], ll[kMergeU], oo[kMergeU];
        #pragma unroll
        for (int k = 0; k < kMergeU; ++k) {        /* hoisted ahead of the skip-select:
                                                    * one dependency-free load wave */
            if (sp0 + k < splits) {
                mm[k] = psm[base + sp0 + k];
                ll[k] = psl[base + sp0 + k];
                oo[k] = pso[(base + sp0 + k) * D + d];
            }
        }
        #pragma unroll
        for (int k = 0; k < kMergeU; ++k)          /* the accumulate is verbatim v153 */
            if (sp0 + k < splits && mm[k] != -INFINITY) {
                const float w = exp2f(mm[k] - mx);
                num += w * oo[k];
                den += w * ll[k];
            }
    }
    const float o = (den > 0.0f ? num * (1.0f / den) : 0.0f) * vds;
    if (P.out_f32) ((float*)P.out)[ob] = o;
    else           ((uint16_t*)P.out)[ob] = f32_bf16_w(o);
}

/* Q46 split-count policy arms. ONE binary carries both:
 *   R3D_SPLIT_POLICY=0  the fixed closed form below, verbatim (the 2026-10-24 retune);
 *   R3D_SPLIT_POLICY=1  wave-efficiency policy (FA2 rule, QFN pass3 P3-009).
 * Default = R3D_SPLIT_POLICY_DEFAULT (set by the Q46 A/B verdict, see the report).
 *
 * The wave rule (host-only arithmetic; the arena hook and the launcher run the SAME
 * function so the declared arena always covers what the launch asks for -- the arena
 * law is structural, not hoped-for):
 *   units  = heads_per_block * max(n_seq,1)  (q_heads plain, kv_heads folded, as today)
 *   blocks(s) = units * s
 *   short-circuit: the FIRST s whose blocks(s) reaches 0.8*CUs (FA2: stop splitting
 *     once there are enough blocks to fill the device); if units alone already
 *     reaches it, that is s=1 -> no split arm at all, exactly the FA2 meaning.
 *   else: wave_eff(s) = blocks / (CUs * ceil(blocks/CUs)) over s <= smax; pick the
 *     SMALLEST s whose eff is within 0.85 of the best.
 *   caps identical to the closed form (tiles_kv, tiles_kv/2 for plain, 64 -- the 64
 *   cap is what preserves the arena sizing law and the bench's 64-slot scratch).
 * CU count is queried from the device once per process (hipGetDeviceProperties'
 * multiProcessorCount). On-card fact 2026-10-27, this XTX: the HIP runtime answers
 * 48 while rocminfo lists 96 "Compute Unit" -- RDNA3 HIP multiProcessorCount counts
 * 16-lane SIMD pairs, not the 96 physical CUs. The runtime value is what the wave
 * model uses (a wave is the device's own accounting), the sweep A/B's both divisors
 * via R3D_SPLIT_CU (the explicit override), and 48 is the query-failure fallback so
 * the dead path matches the device's live answer.
 * Precision class of arm 1: it CHANGES SPLIT COUNTS, which changes the partial-combine
 * order -- precision-visible on split-sensitive attention rows (the m=-inf inert law
 * covers degenerate splits, NOT the ordering): graded by battery tolerance + a decode
 * hash sweep, not by zero-diff. Arm 0 is bit-verbatim by construction. */
#define R3D_SPLIT_POLICY_DEFAULT 0

int decode_split_policy(void) {
    static int arm = -1;
    if (arm < 0) {
        const char* e = getenv("R3D_SPLIT_POLICY");
        arm = e ? (atoi(e) != 0 ? 1 : 0) : R3D_SPLIT_POLICY_DEFAULT;
    }
    return arm;
}

int decode_split_cu(void) {
    static int cu = -1;
    if (cu < 0) {
        const char* e = getenv("R3D_SPLIT_CU");
        if (e && atoi(e) > 0) {
            cu = atoi(e);
        } else {
            hipDeviceProp_t d;
            int dev = 0;
            cu = (hipGetDevice(&dev) == hipSuccess &&
                  hipGetDeviceProperties(&d, dev) == hipSuccess &&
                  d.multiProcessorCount > 0) ? (int)d.multiProcessorCount : 48;
        }
    }
    return cu;
}

/* The shape law, host side, one closed form shared by the launch and the arena hook.
 * tiles_kv: KV tile upper bound (max_ctx or the block table width, kBN keys each).
 * The plain arm's 192: two waves per CU on 96 CUs -- slices want to be long enough
 * that partial traffic (written once, read once) stays noise next to the KV bytes (it
 * is ~1% at the sizes this row serves), and short enough that every CU carries a
 * block. The fold arm's 128 (and tiles_kv cap): the retune constant from the
 * 2026-10-24 verify-band sweep, see the header law note -- 1.3x+ better wherever the
 * old two-wave target resolved above it, identical where the cap already pinned lower.
 * The head count passed in is q_heads for the plain arm and kv_heads for the GQA-fold
 * arm -- folded blocks are gqa-times fewer, so the same wave target needs gqa-times
 * more slices (and the fold arm is the more demanding arena consumer). */
int decode_splits(int64_t q_heads, int64_t n_seq, int64_t tiles_kv, int forced,
                  bool fold) {
    if (forced > 0) return forced > 64 ? 64 : forced;
    const int64_t units = q_heads * (n_seq > 0 ? n_seq : 1);
    const int64_t cap = fold ? tiles_kv : tiles_kv / 2;
    int64_t smax = tiles_kv < 64 ? tiles_kv : 64;
    if (cap < smax) smax = cap;
    if (smax < 1) smax = 1;

    int64_t s;
    if (decode_split_policy() == 0) {                    /* fixed closed form */
        s = (fold ? 128 : 192) / units;
    } else {                                             /* wave-efficiency policy */
        const int cu = decode_split_cu();
        const int64_t fill = (int64_t)cu * 8 / 10;       /* 0.8*CUs, integer */
        s = 0;
        double best = 0.0;
        for (int64_t t = 1; t <= smax; ++t) {
            const int64_t blocks = units * t;
            if (blocks >= fill) { s = t; break; }        /* FA2 short-circuit */
            const int64_t waves = (blocks + cu - 1) / cu;
            const double eff = (double)blocks / (double)(waves * cu);
            if (eff > best) best = eff;
        }
        if (s <= 0) {                                    /* cap never reached fill */
            for (int64_t t = 1; t <= smax; ++t) {
                const int64_t blocks = units * t;
                const int64_t waves = (blocks + cu - 1) / cu;
                const double eff = (double)blocks / (double)(waves * cu);
                if (eff >= 0.85 * best) { s = t; break; }
            }
        }
        if (s <= 0) s = 1;
    }
    if (s > smax) s = smax;
    if (s < 1) s = 1;
    static int once = 0;
    if (!once && getenv("R3D_DEBUG_DECODE")) {           /* liveness: names the arm */
        once = 1;
        printf("DBG attn-decode split-policy arm=%s cu=%lld units=%lld tiles_kv=%lld "
               "fold=%d s=%lld\n",
               decode_split_policy() ? "wave" : "fixed",
               (long long)decode_split_cu(), (long long)units, (long long)tiles_kv,
               (int)fold, (long long)s);
    }
    return (int)s;
}

/* Bytes of partials for one launch: rows*q_heads*splits*(head_dim+2) floats.
 * (rows*q_heads is the (token, q-head) row space -- the fold arm writes the SAME
 * index space, one block just now fills gqa of those head slots per tile row.) */
int64_t decode_partial_bytes(int64_t rows, int64_t q_heads, int splits, int64_t head_dim) {
    return rows * q_heads * (int64_t)splits * (head_dim + 2) * 4;
}

/* Can this shape take the GQA fold? Any gqa > 1: the tile's row space is folded
 * rows (token-major across the kv group), and tiles_q covers them with the loose
 * host bound (P.rows*gqa/kBM) -- the body bounds per-seq against qlen*gqa exactly
 * and dead tiles exit at the door. That holds for ragged batches too: the engine
 * always passes cu_seqlens on decode (live DBG proved the older !cus gate killed
 * the fold for every serve call), and per-seq lengths never need to be known
 * host-side -- only the tile grid bound does, and the loose one is exact for the
 * n_seq==1 case serve actually runs. */
bool decode_fold_ok(int64_t gqa) {
    return gqa > 1;
}

} /* anonymous namespace */

/* ---- arena hook. Declare-time calls carry parameters and no tensors (the batch does
 * not exist yet): n_head, max_seqs, max_ctx and the band's q_len upper bound size the
 * worst step over every seq count 1..max_seqs (splits*seqs is not monotone in seqs; the
 * law is a closed form, so the sweep is free). No max_ctx and no tensors: answer 0 --
 * the entry then runs the SPLITK=0 tile arm, which needs no arena. A never-abort answer
 * on purpose: an under-sized arena must cost a slower path, never a dead engine. */
extern "C" int64_t r3d_rad_attn_decode_scratch(const RadArgs* a) {
    const int sp_param = (int)rad_args_geti_or(a, "splits", 0);
    const int64_t n_head   = rad_args_geti_or(a, "n_head", 0);
    const int64_t max_seqs = rad_args_geti_or(a, "max_seqs", 0);
    const int64_t D        = rad_args_geti_or(a, "head_dim", 0);
    if (n_head > 0 && max_seqs > 0) {
        if (D != 64 && D != 128) {
            /* 256 claims an arena ONLY while the R3D_HD256 lever is on (it is by
             * default since the 2026-10-27 accept): with R3D_HD256=0 the decode row
             * never launches a 256 split, and asking the engine to carve an arena
             * the model class cannot use would cost real KV budget for nothing
             * (issue #11 m3). Same law as both launchers -- r3d_hd256_on(). */
            if (D != 256 || !r3d_hd256_on()) return 0;
        }
        int64_t mc = rad_args_geti_or(a, "max_ctx", 0);
        if (mc <= 0) return 0;
        const int64_t gqa  = rad_args_geti_or(a, "gqa", 1) > 0 ? rad_args_geti_or(a, "gqa", 1) : 1;
        const int64_t kvh  = gqa > 1 ? n_head / gqa : n_head;
        const int64_t qlen = rad_args_geti_or(a, "q_len", 8) > 0 ? rad_args_geti_or(a, "q_len", 8) : 8;
        const int64_t tkv  = (mc + kBN - 1) / kBN;
        const bool foldable = decode_fold_ok(gqa);
        int64_t worst = 0;
        for (int64_t S = 1; S <= max_seqs; ++S) {
            int sp = decode_splits(n_head, S, tkv, sp_param, false);
            if (foldable) {
                const int spf = decode_splits(kvh, S, tkv, sp_param, true);
                if (spf > sp) sp = spf;                       /* arm-agnostic worst case */
            }
            const int64_t need = decode_partial_bytes(S * qlen, n_head, sp, D);
            if (need > worst) worst = need;
        }
        return worst;
    }

    R3dAttnParams P{};
    if (r3d_attn_fill(a, P) == RAD_OK && P.head_dim > 0) {
        int64_t mc = rad_args_geti_or(a, "max_ctx", 0);
        if (mc <= 0) mc = P.jcap;
        const int64_t tkv = (mc + kBN - 1) / kBN;
        int sp = decode_splits(P.q_heads, P.n_seq, tkv, sp_param, false);
        if (decode_fold_ok(P.gqa)) {
            const int spf = decode_splits(P.kv_heads, P.n_seq, tkv, sp_param, true);
            if (spf > sp) sp = spf;
        }
        return decode_partial_bytes(P.rows, P.q_heads, sp, P.head_dim);
    }
    return 0;
}

/* ---- entry: the row the engine resolves for the decode band; must serve what it
 * resolves for (no declines -- every refusal of the WMMA path is served by the scalar
 * port or the SPLITK=0 tile, in this entry, on the same filled scalars). ---- */
extern "C" int r3d_rad_attn_paged_decode(const RadArgs* a, RadStream s) {
    R3dAttnParams P{};
    const int rc = r3d_attn_fill(a, P);
    if (rc != RAD_OK) return rc;          /* a shape no arm serves: the honest error v1
                                             * would have returned, from the same lines */
    hipStream_t stream = (hipStream_t)(uintptr_t)s;

    static int allowed = -1;
    if (allowed < 0) {
        hipDeviceProp_t d;
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&d, dev) != hipSuccess)
            allowed = 0;
        else
            allowed = (strstr(d.gcnArchName, "gfx1100") || strstr(d.gcnArchName, "gfx1101") ||
                       strstr(d.gcnArchName, "gfx1102") || strstr(d.gcnArchName, "gfx1103")) ? 1 : 0;
        if (getenv("R3D_NO_WMMA")) allowed = 0;   /* one switch for every WMMA row */
    }

    const int dbg = getenv("R3D_DEBUG_DECODE") ? 1 : 0;

    /* The tile arm: SPLITK=0 through the prefill row's launcher -- the exact kernel the
     * old row ran. Used when splits resolves to 1, when the arena is short, and as the
     * WMMA-chip fallback of last resort; v1 under a foreign chip or R3D_NO_WMMA. */
    static int v2 = -1;
    if (v2 < 0) v2 = getenv("R3D_ATTN_V2") ? 1 : 0;      /* v1 default: see wmma launcher */
    /* R3D_HD256 (issue #11 milestone 3): head_dim 256 joins the decode row's split-KV
     * family -- the same fitted body at SPLITK=1 (partial store and slice cut are
     * D-generic, see the body's m3 note), merged by the same D-generic merge kernel.
     * v2/v3 and fp8 KV at 256 have no instantiation (the prefill launcher's law,
     * mirrored here): those shapes keep d256=0, take tile_or_v1, and land on the m2
     * tile or v1 exactly as before this row changed. R3D_HD256=0 (the withdrawal,
     * q8x value law): every shape sees the identical gate it saw at v148. Accepted
     * 2026-10-27: the lever defaults ON -- the engine reaches this entry for 256
     * decode shapes through the decode mirror (r3d_attn_paged.hip) and the
     * attn_paged_decode_bf16kv_h256 band row (r3d_rows.cpp). */
    const int hd256 = r3d_hd256_on();
    const int d256 = hd256 && P.head_dim == 256 && !v2 && P.kvdt == RAD_BF16;
    const int wmma_ok = allowed && P.qdt == RAD_BF16
                     && (P.kvdt == RAD_BF16 || P.kvdt == RAD_F8E4M3)
                     && (P.head_dim == 64 || P.head_dim == 128 || d256);
    const auto tile_or_v1 = [&]() -> int {
        if (wmma_ok && r3d_attn_paged_wmma_launch(P, stream) == RAD_OK) return RAD_OK;
        return r3d_attn_v1_launch(P, stream);
    };
    if (!wmma_ok) return tile_or_v1();

    int forced = (int)rad_args_geti_or(a, "splits", 0);
    if (const char* e = getenv("R3D_FORCE_SPLITS")) {   /* bench knob: pin the split count */
        const int v = atoi(e);
        if (v > 0) forced = v;
    }
    if (getenv("R3D_NO_SPLITK")) forced = 1;             /* A/B kill: the tile, same row */
    static int fold = -1;
    if (fold < 0) fold = getenv("R3D_NO_GQA") ? 0 : 1;   /* A/B kill: the GQA fold arm */
    int64_t mc = rad_args_geti_or(a, "max_ctx", 0);
    if (mc <= 0) mc = P.jcap;
    const int64_t tkv = (mc + kBN - 1) / kBN;
    const int spl_plain = decode_splits(P.q_heads, P.n_seq, tkv, forced, false);
    /* GQA fold (llama.cpp's tile-kernel fold, ported): one block per KV head streams
     * the slice once for the whole q-head group, so decode KV traffic drops by gqa.
     * Folded blocks are gqa-times fewer, so its splits law runs on kv_heads; a shape
     * the fold gate refuses, or one where fold resolves to 1 slice, takes the plain
     * split arm unchanged (only gqa==1 ever resolves that way). */
    const bool fold_ok = fold && !v2 && decode_fold_ok(P.gqa);
    const int spl_fold = fold_ok ? decode_splits(P.kv_heads, P.n_seq, tkv, forced, true) : 1;
    const bool use_fold = fold_ok && spl_fold > 1;
    const int spl = use_fold ? spl_fold : spl_plain;
    if (spl <= 1) return tile_or_v1();

    const int64_t per_seq = (P.cus ? P.rows : P.q_len_uniform) * (use_fold ? P.gqa : 1);
    const int64_t tiles_q = (per_seq + kBM - 1) / kBM;   /* per-seq exact, ragged loose */
    const int64_t gz = tiles_q * spl;
    const int64_t gx = use_fold ? P.kv_heads : P.q_heads;
    if (gx > 65535 || P.q_heads > 65535 || P.n_seq > 65535 || gz <= 0 || gz > 65535
        || P.rows > 65535)
        return tile_or_v1();

    const int64_t need = decode_partial_bytes(P.rows, P.q_heads, spl, P.head_dim);
    if (!a->scratch || a->scratch_bytes < need) {
        if (dbg) printf("DBG attn-decode arena %lld < need %lld: tile arm\n",
                        (long long)a->scratch_bytes, (long long)need);
        return tile_or_v1();
    }

    P.pf = a->scratch;
    P.splits = spl;

    /* v1 slice kernel is the default -- the v2 256-thread schedule measured slower at
     * deep ctx on the prefill bench (bit-identical, 0.43-0.80x; see the wmma launcher).
     * R3D_ATTN_V2=1 covers BOTH halves of this row: the tile_or_v1 arm through the
     * prefill launcher's own v2 selection, and the slice kernel here. The merge kernel
     * is body-agnostic -- same arena contract -- so one instantiation serves both.
     * (v2's flag is read at the top of this entry: the fold gate refuses v2 shapes.) */

    #define R3D_DK_LAUNCH(K, D, KV)                                                                  \
        do {                                                                                         \
            hipLaunchKernelGGL((K<D, KV, 1>),                                                        \
                               dim3((unsigned)P.q_heads, (unsigned)P.n_seq, (unsigned)gz),           \
                               dim3(v2 ? kT : kThreads), 0, stream, P);                             \
            hipError_t e0 = hipGetLastError();                                                       \
            if (e0 != hipSuccess) return RAD_E_DEVICE;   /* partial state: fail the step; no arm\n    * may resume on half a launch -- libr4d's decode entries abort the same way */             \
            hipLaunchKernelGGL((attn_splitk_merge_kernel<D>),                                        \
                               dim3((unsigned)P.q_heads, (unsigned)P.rows), dim3(D), 0, stream, P, spl);\
            hipError_t e1 = hipGetLastError();                                                       \
            return e1 == hipSuccess ? RAD_OK : RAD_E_DEVICE;                                         \
        } while (0)

    if (dbg) printf("DBG attn-decode splitk hd=%lld kvdt=%d spl=%d fold=%d tiles_kv=%lld "
                    "grid(%lld,%lld,%lld) merge(%lld,%lld) need=%lld\n",
                    (long long)P.head_dim, (int)P.kvdt, spl, (int)use_fold, (long long)tkv,
                    (long long)gx, (long long)P.n_seq, (long long)gz,
                    (long long)P.q_heads, (long long)P.rows, (long long)need);

    const int kvbf = (P.kvdt == RAD_BF16);
    const int d128 = (P.head_dim == 128);
    /* DBUF slice arm (see the prefill launcher's comment): default-on after the
     * paired bench; R3D_ATTN_NO_DBUF=1 forces the old staging arm. The merge
     * kernel is schedule-agnostic -- same arena contract -- no second arm. */
    static int dbuf = -1;
#ifdef R3D_DBG
    dbuf = 0;
#else
    if (dbuf < 0) dbuf = getenv("R3D_ATTN_NO_DBUF") ? 0 : 1;
#endif
    /* Q10 fp8 staging arms (stage_kv_tile's F8ST slot). DEFAULT 3 (LUT + wide
     * gather): the 2026-10-25 A/B (3 interleaved rounds, hash-equal, disjoint
     * spreads) put st3 +33..+154% over v129, st2 +8..11%, st1 +3..6%. R3D_F8ST
     * overrides: 0 verbatim withdrawal, 1 branchless VALU, 2 LUT scalar. Read
     * once per process. R3D_DBG forces verbatim: the dbg wire keeps its own
     * instrumented scalar loop that bypasses stage_kv_tile (same law as DBUF). */
    static int f8st = -1;
#ifdef R3D_DBG
    f8st = 0;
#else
    if (f8st < 0) { const char* sv = getenv("R3D_F8ST"); f8st = sv ? atoi(sv) : 3;
                    if (f8st < 0 || f8st > 3) f8st = 3; }
#endif
    /* Q11 (R3D_ATT_FS): fragment-side softmax, DEFAULT ON since build_v142 (see
     * r3d_attn_paged_wmma.hip for the gate evidence); R3D_ATT_FS=0 withdraws to
     * the v140 sweep arm. DBUF bf16 arms only, fold and plain; the split-KV merge
     * below reads the same m/l arena words. */
    static int fs = -1;
#ifdef R3D_DBG
    fs = 0;
#else
    if (fs < 0) { const char* sv = getenv("R3D_ATT_FS"); fs = (sv && atoi(sv) == 0) ? 0 : 1; }
#endif
    const bool dbuf_ok = dbuf && !v2 && kvbf && (P.kv_sd == 1) && ((((uintptr_t)P.kv) & 15) == 0);
    #define R3D_DK_LAUNCH4(K, D, KV, SK, DB, FD)                                                           \
        do {                                                                                               \
            hipLaunchKernelGGL((K<D, KV, SK, DB, FD>),                                                     \
                               dim3((unsigned)gx, (unsigned)P.n_seq, (unsigned)gz),                        \
                               dim3(kThreads), 0, stream, P);                                             \
            hipError_t e0 = hipGetLastError();                                                             \
            if (e0 != hipSuccess) return RAD_E_DEVICE;   /* partial state: fail the step */                \
            hipLaunchKernelGGL((attn_splitk_merge_kernel<D>),                                              \
                               dim3((unsigned)P.q_heads, (unsigned)P.rows), dim3(D), 0, stream, P, spl);   \
            hipError_t e1 = hipGetLastError();                                                             \
            return e1 == hipSuccess ? RAD_OK : RAD_E_DEVICE;                                               \
        } while (0)
    #define R3D_DK_ARMS(K)                                                           \
        if      (d128 &&  kvbf) R3D_DK_LAUNCH(K, 128, RAD_BF16);                     \
        else if (d128 && !kvbf) R3D_DK_LAUNCH(K, 128, RAD_F8E4M3);                   \
        else if (!d128 &&  kvbf) R3D_DK_LAUNCH(K, 64, RAD_BF16);                     \
        else                     R3D_DK_LAUNCH(K, 64, RAD_F8E4M3);
    /* Q10 armed fp8 staging arms: same launch shapes as DK_LAUNCH/DK_LAUNCH4,
     * F8ST template slot filled. Only attn_paged_wmma_kernel (v1) is armed --
     * kernel2/kernel3 keep their own 3-arg templates and their own macro arms. */
    #define R3D_DK_LAUNCH8(K, D, FD, ST)                                                                   \
        do {                                                                                               \
            hipLaunchKernelGGL((K<D, RAD_F8E4M3, 1, 0, FD, 0, ST>),                                        \
                               dim3((unsigned)((FD) ? gx : P.q_heads), (unsigned)P.n_seq, (unsigned)gz),   \
                               dim3(kThreads), 0, stream, P);                                              \
            hipError_t e0 = hipGetLastError();                                                             \
            if (e0 != hipSuccess) return RAD_E_DEVICE;                                                     \
            hipLaunchKernelGGL((attn_splitk_merge_kernel<D>),                                              \
                               dim3((unsigned)P.q_heads, (unsigned)P.rows), dim3(D), 0, stream, P, spl);   \
            hipError_t e1 = hipGetLastError();                                                             \
            return e1 == hipSuccess ? RAD_OK : RAD_E_DEVICE;                                               \
        } while (0)
    #define R3D_DK_F8(K, D, FD)                    \
        if      (f8st == 1) R3D_DK_LAUNCH8(K, D, FD, 1); \
        else if (f8st == 2) R3D_DK_LAUNCH8(K, D, FD, 2); \
        else if (f8st == 3) R3D_DK_LAUNCH8(K, D, FD, 3)
    /* Q11 FS arm: DBUF + fragment-side softmax (dbuf_ok implies bf16). Slot order
     * <D, KVDT, SPLITK=1, DBUF=1, FOLD=FD, PAD=0, F8ST=0, SB=0, FS=1>; same merge. */
    #define R3D_DK_LAUNCHFS(K, D, FD)                                                            \
        do {                                                                                               \
            hipLaunchKernelGGL((K<D, RAD_BF16, 1, 1, FD, 0, 0, 0, 1>),                                     \
                               dim3((unsigned)((FD) ? gx : P.q_heads), (unsigned)P.n_seq, (unsigned)gz),   \
                               dim3(kThreads), 0, stream, P);                                             \
            hipError_t e0 = hipGetLastError();                                                             \
            if (e0 != hipSuccess) return RAD_E_DEVICE;   /* partial state: fail the step */                \
            hipLaunchKernelGGL((attn_splitk_merge_kernel<D>),                                              \
                               dim3((unsigned)P.q_heads, (unsigned)P.rows), dim3(D), 0, stream, P, spl);   \
            hipError_t e1 = hipGetLastError();                                                             \
            return e1 == hipSuccess ? RAD_OK : RAD_E_DEVICE;                                               \
        } while (0)
    /* Milestone 3 (issue #11): the 256 slice arm. Same fitted body as the m2 tile at
     * SPLITK=1, so it inherits the dynamic-LDS law: the attribute is raised once per
     * instantiation and the launch passes the launcher-side byte count (the kernel's
     * static_assert pins r3d_attn256_smem(PAD) to its own C::total). PAD/DBUF/FOLD are
     * literal template slots selected by the same env/default law as the tile arm
     * (R3D_HD256_PAD default 1, DBUF behind the vec gate, fold behind the row's
     * existing fold gate). A failed attribute falls back to tile_or_v1 -- the tile can
     * lose a step to a driver limit, never cost the engine one. No FS slot: fragment-
     * side softmax is not yet instantiated for 256 (same law as the m2 tile). The
     * merge kernel needs no dynamic LDS, so it launches with 0 as always. */
    /* Q45a lever: R3D_ATTN256_STAGEBAR (env, default 0). 0 = shipped full-drain
     * staging barriers verbatim (withdrawal arm, byte-identical); 1 = lgkmcnt-scoped
     * variant at the 0x4c00-class staging sites of the DBUF branch (see
     * r3d_attn_wmma_body.inc). Doubles only the DBUF=1 h256 instantiations. */
    #define R3D_DK_L256(DB, PAD, FD)                                                                     \
        do {                                                                                             \
            if (DB && stagebar256) { R3D_DK_L256_S(DB, PAD, FD, 1) }                                     \
            else                   { R3D_DK_L256_S(DB, PAD, FD, 0) }                                     \
        } while (0)
    #define R3D_DK_L256_S(DB, PAD, FD, SB)                                                               \
        do {                                                                                             \
            static int attr = -1;                                                                        \
            if (attr < 0)                                                                                \
                attr = hipFuncSetAttribute(                                                              \
                           (const void*)&(attn_paged_wmma_kernel<256, RAD_BF16, 1, DB, FD, PAD, 0, 0, 0, SB>),\
                           hipFuncAttributeMaxDynamicSharedMemorySize,                                   \
                           r3d_attn256_smem(PAD)) == hipSuccess ? 1 : 0;                                 \
            if (!attr) return tile_or_v1();                                                              \
            static int once = 0;                                                                         \
            if (dbg && !once) {                                                                          \
                once = 1;                                                                                \
                printf("DBG attn-decode hd256 slice launch pad=%d dbuf=%d fold=%d smem=%d spl=%d "       \
                       "grid(%lld,%lld,%lld) merge(%lld,%lld) need=%lld\n",                              \
                       PAD, DB, FD, r3d_attn256_smem(PAD), spl,                                         \
                       (long long)gx, (long long)P.n_seq, (long long)gz,                                 \
                       (long long)P.q_heads, (long long)P.rows, (long long)need);                        \
            }                                                                                            \
            if (!r3d_256_arm_logged) {                                                                   \
                r3d_256_arm_logged = 1;                                                                  \
                fprintf(stderr, "DBG attn-decode hd256 arm launched pad=%d dbuf=%d fold=%d "             \
                        "stagebar=%d\n", PAD, DB, FD, SB);                                               \
            }                                                                                            \
            hipLaunchKernelGGL((attn_paged_wmma_kernel<256, RAD_BF16, 1, DB, FD, PAD, 0, 0, 0, SB>),     \
                               dim3((unsigned)gx, (unsigned)P.n_seq, (unsigned)gz),                      \
                               dim3(kThreads), r3d_attn256_smem(PAD), stream, P);                        \
            hipError_t e0 = hipGetLastError();                                                           \
            if (e0 != hipSuccess) return RAD_E_DEVICE;   /* partial state: fail the step */              \
            hipLaunchKernelGGL((attn_splitk_merge_kernel<256>),                                          \
                               dim3((unsigned)P.q_heads, (unsigned)P.rows), dim3(256), 0, stream, P, spl);\
            hipError_t e1 = hipGetLastError();                                                           \
            return e1 == hipSuccess ? RAD_OK : RAD_E_DEVICE;                                             \
        } while (0)
    if (d256) {
        static int pad256 = -1;
        if (pad256 < 0) pad256 = getenv("R3D_HD256_PAD") ? atoi(getenv("R3D_HD256_PAD")) : 1;
        static int stagebar256 = -1;   /* Q45a: unset/0 = shipped arm verbatim */
        if (stagebar256 < 0) { const char* e = getenv("R3D_ATTN256_STAGEBAR");
                               stagebar256 = (e && atoi(e) != 0) ? 1 : 0; }
        static int r3d_256_arm_logged = 0;  /* Q45a liveness (stderr law, window 45) */
        /* FD/DB/PAD reach the template as literals (two-level dispatch: the runtime
         * gates pick which of the eight instantiations to emit). */
        if (use_fold) {
            if      (dbuf_ok && pad256) R3D_DK_L256(1, 1, 1);
            else if (dbuf_ok)           R3D_DK_L256(1, 0, 1);
            else if (pad256)            R3D_DK_L256(0, 1, 1);
            else                        R3D_DK_L256(0, 0, 1);
        }
        else {
            if      (dbuf_ok && pad256) R3D_DK_L256(1, 1, 0);
            else if (dbuf_ok)           R3D_DK_L256(1, 0, 0);
            else if (pad256)            R3D_DK_L256(0, 1, 0);
            else                        R3D_DK_L256(0, 0, 0);
        }
    }
    if (v2)           { R3D_DK_ARMS(attn_paged_wmma_kernel2) }
    else if (use_fold) {
        /* GQA fold: one block per kv head (grid x = gx = kv_heads), rows = the group's
         * heads token-major. DBUF staging composes (bf16 arm only, dispatch on the
         * runtime gate -- template args must stay literal); fp8 keeps the scalar
         * staging. Per-row SOURCE arithmetic is v1's exactly, but pinned-splits
         * hash equality with the plain arm is a PARITY-CLASS law, not a bit law
         * (Q53, branch card/q53-fold-fnv forensics): v_wmma_f32_16x16x16_bf16
         * dot results are lane-parity stable, not arm-stable. The fold puts a row
         * at tile row = head-in-group (the token selects the wave); the plain arm
         * puts tokens on tile rows. On gfx1100 the hardware dot chain returns
         * values differing by 1-2 f32 ulp between even and odd tile-row positions
         * for IDENTICAL operand bytes -- witnessed inside one arm by
         * tools/q53_probe.cpp Q53_DUPROWS (even-parity tokens agree bit for bit,
         * odd-parity ones do not), so no source path reassociates and no source
         * path can pin both arms to the same lane parity: packing IS the fold.
         * Hence drift(tok, head) iff parity(tok) != parity(head-in-group): at
         * most 1 bf16 ulp per output element (or a difference <= 2e-6 in absolute
         * value at the near-zero softmax tails), under ~0.5% of elements, both
         * arms inside the oracle
         * class. Pinned-splits hashes stay a FREE bit gate for same-parity levers
         * (any arm against itself, schedule or staging changes); fold-vs-plain is
         * a class gate -- assert it with tools/q53_law_check.sh. */
        if      (dbuf_ok && fs && d128) R3D_DK_LAUNCHFS(attn_paged_wmma_kernel, 128, 1);
        else if (dbuf_ok && fs)         R3D_DK_LAUNCHFS(attn_paged_wmma_kernel,  64, 1);
        else if (dbuf_ok &&  d128) R3D_DK_LAUNCH4(attn_paged_wmma_kernel, 128, RAD_BF16,   1, 1, 1);
        else if (dbuf_ok)          R3D_DK_LAUNCH4(attn_paged_wmma_kernel, 64,  RAD_BF16,   1, 1, 1);
        else if (d128 &&  kvbf)    R3D_DK_LAUNCH4(attn_paged_wmma_kernel, 128, RAD_BF16,   1, 0, 1);
        else if (d128 && !kvbf)    { R3D_DK_F8(attn_paged_wmma_kernel, 128, 1);
                                     else R3D_DK_LAUNCH4(attn_paged_wmma_kernel, 128, RAD_F8E4M3, 1, 0, 1); }
        else if (!d128 &&  kvbf)   R3D_DK_LAUNCH4(attn_paged_wmma_kernel, 64,  RAD_BF16,   1, 0, 1);
        else                       { R3D_DK_F8(attn_paged_wmma_kernel, 64, 1);
                                     else R3D_DK_LAUNCH4(attn_paged_wmma_kernel, 64, RAD_F8E4M3, 1, 0, 1); }
    }
    else if (dbuf_ok && fs) {
        if      (d128)  R3D_DK_LAUNCHFS(attn_paged_wmma_kernel, 128, 0);
        else            R3D_DK_LAUNCHFS(attn_paged_wmma_kernel,  64, 0);
    }
    else if (dbuf_ok) {
        if      (d128 &&  kvbf) R3D_DK_LAUNCH4(attn_paged_wmma_kernel, 128, RAD_BF16, 1, 1, 0);
        else if (d128 && !kvbf) R3D_DK_LAUNCH4(attn_paged_wmma_kernel, 128, RAD_F8E4M3, 1, 0, 0);
        else if (!d128 &&  kvbf) R3D_DK_LAUNCH4(attn_paged_wmma_kernel, 64, RAD_BF16, 1, 1, 0);
        else                     R3D_DK_LAUNCH4(attn_paged_wmma_kernel, 64, RAD_F8E4M3, 1, 0, 0);
    }
    else if (f8st && !kvbf) {            /* v1 plain split-K, armed (v2 caught above) */
        if (d128) { R3D_DK_F8(attn_paged_wmma_kernel, 128, 0);
                    else R3D_DK_LAUNCH(attn_paged_wmma_kernel, 128, RAD_F8E4M3); }
        else      { R3D_DK_F8(attn_paged_wmma_kernel, 64, 0);
                    else R3D_DK_LAUNCH(attn_paged_wmma_kernel, 64, RAD_F8E4M3); }
    }
    else              { R3D_DK_ARMS(attn_paged_wmma_kernel)  }
    #undef R3D_DK_ARMS
    #undef R3D_DK_LAUNCH4
    #undef R3D_DK_LAUNCH8
    #undef R3D_DK_F8
    #undef R3D_DK_LAUNCH
    return RAD_E_DEVICE;   /* unreachable: every macro arm returns */
}
