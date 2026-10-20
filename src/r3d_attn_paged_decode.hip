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
 * A slice with no tiles writes m=-inf, l=0 and no O bytes; the merge skips the O read on
 * m==-inf, so the inert splits cost two floats, not a row.
 *
 * Shape law (host, one place): splits = min(192 / (q_heads*n_seq), tiles_kv/2,
 * tiles_kv, 64), the 192 a two-wave-per-CU target for 96 CUs; an explicit `splits`
 * (the row's tuner axis, or the engine passing a tuned value) wins outright. The SAME
 * closed form sizes the arena at declare (swept over every seq count up to max_seqs,
 * because splits*seqs is not monotone in seqs).
 *
 * Fallbacks, all inside the row (a resolved row must serve, so nothing here declines
 * the op -- only the arithmetic path changes): foreign chip or R3D_NO_WMMA -> the scalar
 * reference port; splits resolving to 1, or an arena smaller than the law needs -> the
 * SPLITK=0 tile; both of those never launch the split kernel, so an untouched partial
 * buffer is not a correctness question. Same scalars every way: one r3d_attn_fill call.
 */

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rad_abi.h"
#include "rad_plugin.h"
#include "r3d_attn.h"

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
template <int D, int KVDT, int SPLITK, int DBUF = 0>
__global__ __launch_bounds__(kThreads) void attn_paged_wmma_kernel(R3dAttnParams) {}
template <int D, int KVDT, int SPLITK>
__global__ __launch_bounds__(kT) void attn_paged_wmma_kernel2(R3dAttnParams) {}
} /* anonymous namespace */

#endif

namespace {

/* ---- merge: one output row per block, one thread per head_dim column ----
 * out = vds * sum_sp w_sp * O_sp / sum_sp w_sp * l_sp,  w_sp = exp2(m_sp - max m),
 * inert splits (m=-inf) skipped with no O traffic. m/l addresses are warp-uniform, so
 * the skip predicate cannot diverge a wave. With one live split this is the tile's own
 * (O * (1/l)) * vds, to the rounding. */
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
    for (int sp = 0; sp < splits; ++sp) {
        const float m = psm[base + sp];
        if (m > mx) mx = m;
    }
    if (mx == -INFINITY) {                  /* every split inert: the zero row, as v1 */
        if (P.out_f32) ((float*)P.out)[ob] = 0.0f;
        else           ((uint16_t*)P.out)[ob] = 0;
        return;
    }
    float num = 0.0f, den = 0.0f;
    for (int sp = 0; sp < splits; ++sp) {
        const float m = psm[base + sp];
        if (m == -INFINITY) continue;
        const float w = exp2f(m - mx);
        num += w * pso[(base + sp) * D + d];
        den += w * psl[base + sp];
    }
    const float o = (den > 0.0f ? num * (1.0f / den) : 0.0f) * vds;
    if (P.out_f32) ((float*)P.out)[ob] = o;
    else           ((uint16_t*)P.out)[ob] = f32_bf16_w(o);
}

/* The shape law, host side, one closed form shared by the launch and the arena hook.
 * tiles_kv: KV tile upper bound (max_ctx or the block table width, kBN keys each).
 * The 192: two waves per CU on 96 CUs -- slices want to be long enough that partial
 * traffic (written once, read once) stays noise next to the KV bytes (it is ~1% at the
 * sizes this row serves), and short enough that every CU carries a block. */
int decode_splits(int64_t q_heads, int64_t n_seq, int64_t tiles_kv, int forced) {
    if (forced > 0) return forced > 64 ? 64 : forced;
    const int64_t units = q_heads * (n_seq > 0 ? n_seq : 1);
    int64_t s = 192 / units;
    const int64_t cap = tiles_kv / 2;
    if (s > cap) s = cap;
    if (s > tiles_kv) s = tiles_kv;
    if (s > 64) s = 64;
    if (s < 1) s = 1;
    return (int)s;
}

/* Bytes of partials for one launch: rows*q_heads*splits*(head_dim+2) floats. */
int64_t decode_partial_bytes(int64_t rows, int64_t q_heads, int splits, int64_t head_dim) {
    return rows * q_heads * (int64_t)splits * (head_dim + 2) * 4;
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
        if (D != 64 && D != 128) return 0;
        int64_t mc = rad_args_geti_or(a, "max_ctx", 0);
        if (mc <= 0) return 0;
        const int64_t gqa  = rad_args_geti_or(a, "gqa", 1) > 0 ? rad_args_geti_or(a, "gqa", 1) : 1;
        (void)gqa;                                   /* the law needs q_heads only */
        const int64_t qlen = rad_args_geti_or(a, "q_len", 8) > 0 ? rad_args_geti_or(a, "q_len", 8) : 8;
        const int64_t tkv  = (mc + kBN - 1) / kBN;
        int64_t worst = 0;
        for (int64_t S = 1; S <= max_seqs; ++S) {
            const int sp = decode_splits(n_head, S, tkv, sp_param);
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
        const int sp = decode_splits(P.q_heads, P.n_seq, tkv, sp_param);
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
    const int wmma_ok = allowed && P.qdt == RAD_BF16
                     && (P.kvdt == RAD_BF16 || P.kvdt == RAD_F8E4M3)
                     && (P.head_dim == 64 || P.head_dim == 128);
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
    int64_t mc = rad_args_geti_or(a, "max_ctx", 0);
    if (mc <= 0) mc = P.jcap;
    const int64_t tkv = (mc + kBN - 1) / kBN;
    const int spl = decode_splits(P.q_heads, P.n_seq, tkv, forced);
    if (spl <= 1) return tile_or_v1();

    const int64_t tiles_q = (P.rows + kBM - 1) / kBM;   /* loose bound, as the tile */
    const int64_t gz = tiles_q * spl;
    if (P.q_heads > 65535 || P.n_seq > 65535 || gz <= 0 || gz > 65535 || P.rows > 65535)
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
     * is body-agnostic -- same arena contract -- so one instantiation serves both. */
    static int v2 = -1;
    if (v2 < 0) v2 = getenv("R3D_ATTN_V2") ? 1 : 0;   /* v1 default: see wmma launcher */

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

    if (dbg) printf("DBG attn-decode splitk hd=%lld kvdt=%d spl=%d tiles_kv=%lld "
                    "grid(%lld,%lld,%lld) merge(%lld,%lld) need=%lld\n",
                    (long long)P.head_dim, (int)P.kvdt, spl, (long long)tkv,
                    (long long)P.q_heads, (long long)P.n_seq, (long long)gz,
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
    const bool dbuf_ok = dbuf && !v2 && kvbf && (P.kv_sd == 1) && ((((uintptr_t)P.kv) & 15) == 0);
    #define R3D_DK_LAUNCH4(K, D, KV, SK, DB)                                                               \
        do {                                                                                               \
            hipLaunchKernelGGL((K<D, KV, SK, DB>),                                                         \
                               dim3((unsigned)P.q_heads, (unsigned)P.n_seq, (unsigned)gz),                \
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
    if (v2)           { R3D_DK_ARMS(attn_paged_wmma_kernel2) }
    else if (dbuf_ok) {
        if      (d128 &&  kvbf) R3D_DK_LAUNCH4(attn_paged_wmma_kernel, 128, RAD_BF16, 1, 1);
        else if (d128 && !kvbf) R3D_DK_LAUNCH4(attn_paged_wmma_kernel, 128, RAD_F8E4M3, 1, 1);
        else if (!d128 &&  kvbf) R3D_DK_LAUNCH4(attn_paged_wmma_kernel, 64, RAD_BF16, 1, 1);
        else                     R3D_DK_LAUNCH4(attn_paged_wmma_kernel, 64, RAD_F8E4M3, 1, 1);
    }
    else              { R3D_DK_ARMS(attn_paged_wmma_kernel)  }
    #undef R3D_DK_ARMS
    #undef R3D_DK_LAUNCH4
    #undef R3D_DK_LAUNCH
    return RAD_E_DEVICE;   /* unreachable: every macro arm returns */
}
