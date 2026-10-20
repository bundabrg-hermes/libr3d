/* r3d_attn_paged_wmma.hip -- WMMA flash tile for attn_paged on RDNA3 (gfx1100 class).

 * Why this TU exists: the two-probe prefill fit (docs/COVERAGE.md, 2026-10-20) puts ~87%
 * of prefill time in attention, and the v1 kernel in r3d_attn_paged.hip runs it at
 * ~133 GFLOP/s -- one workgroup per (row, head), three syncthreads per KEY, and the
 * score dot serialized on thread 0. The card's matrix units were never invited.

 * Shape: one workgroup per (64-query tile, q head, sequence); 128 threads = 4 wave32s,
 * wave w owning query rows [16w, 16w+16). Keys stream in 32-wide tiles (the XTX's KV
 * pages are 16 tokens; a key tile spans at most two block-table entries, gathered per
 * row through the same key_at addressing v1 uses -- unmapped pages mark their columns
 * bad, and bad columns are skipped, never zero-scored, matching ref's key-skip rule).

 * Two passes, not FA2's online rescale: pass one walks K tiles computing S = Q.K^T with
 * WMMA and reduces per-row running max m and sum l; pass two RECOMPUTES the same S tiles
 * (bit-identical: same fragments, same order), forms P = exp2(S*scale2 - m) in bf16, and
 * accumulates O += P.V with WMMA. The reward for 1.5x the QK^T flops is that no WMMA
 * accumulator is ever rescaled mid-loop and no fragment needs a layout conversion --
 * accumulators go LDS->global once, at the end. At the card's tensor-core rate the extra
 * flops are noise; v1's latency was the disease.

 * Arithmetic drift vs libref (adjudicated by kbench rel_l2 + the KLD gate, same contract
 * class as the gemm m64 tile): the WMMA fold order is opaque (reassociation, not rounding
 * class), scores are scaled by the folded scale*kds*log2e instead of kds-per-k and
 * exp-after-max, exp2f replaces (float)exp(double), P narrows to bf16 before the PV
 * product, and v_descale folds at the final multiply instead of per key element. Every
 * one of these is sub-bf16-store rounding; the KLD gate -- built for exactly this
 * question -- is the arbiter, with the v1 kernel as its base arm via R3D_NO_WMMA=1.

 * The exp2 domain: m and l live in S*scale2 with scale2 = scale*kds*log2(e), so
 * w = exp2(S*scale2 - m) == exp(score - max_score) up to the folded-rounding drift above.

 * Serve geometry this tile is built for (MiniCPM5-2B): head_dim 128, 16 q heads / 2 kv
 * heads (gqa 8), block_size 16, bf16 cache. head_dim 64 compiles and runs on the same
 * template; q-f32 or kv-f32 caches, head_dim outside {64,128}, and every refused shape
 * return RAD_E_UNSUPPORTED and v1 serves them, unchanged, forever. Decode rows are
 * eligible too: a qlen=1 sequence wastes 63/64 of a query tile's FMAs but reads the same
 * bytes v1 reads -- decode is bandwidth-bound, and the tensor cores make the waste free
 * (measured, not assumed: the serve benchmark is the gate).

 * Chip gating and the R3D_NO_WMMA kill switch follow r3d_gemm_bf16_m64.hip's lesson: the
 * kernel body compiles behind the gfx1100..gfx1103 gate (rocWMMA's config gate rejects
 * family-generic targets), the LAUNCHER lives outside it and decides by gcnArchName at
 * runtime, so a generic-target build carries a stub the launcher never dispatches.
 */

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rad_abi.h"
#include "r3d_attn.h"

namespace {

/* dtype-widening helpers: bit-identical to the v1 kernel's (same contracts, see
 * r3d_attn_paged.hip); duplicated rather than shared to keep the two TUs independent --
 * these eleven lines are the most-reviewed eleven in the project. */
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
constexpr int kThreads = 128;               /* 4 wave32s, 16 query rows each */
constexpr float kLog2e = 1.44269504088896340736f;

} /* anonymous namespace */

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)

#include <rocwmma/rocwmma.hpp>

namespace {

namespace roCW = rocwmma;
using bf16 = roCW::bfloat16_t;

/* LDS carve, byte offsets into one buffer (16-aligned), per head_dim:
 *   Q   64 x (D+4) bf16   query tile, staged once per workgroup
 *   S   64 x 40    f32    score tile (pad 8: 32 real columns, row-bank spread)
 *   P   64 x 36    bf16   exp'd probabilities, pass two only
 *   KV  32 x (D+4) bf16   key tile, THEN the value tile (aliased: V stages after the
 *                         tile's S-fragments have left the LDS, one sync apart)
 *   meta m/l f32[64], rowbase i64[32], colbad u8[32]
 * O-store staging (32 x (D+8) f32, halves of the output) aliases Q+S after pass two's
 * last fragment read. D=128 total: 40992 B static -- under the 48 KiB static ceiling,
 * no hipFuncSetAttribute dance. */
template <int D> struct Carve {
    static constexpr int kQP = D + 4, kVP = D + 4, kSP = kBN + 8, kPP = kBN + 4;
    static constexpr int kOP = D + 8;
    static constexpr int offQ  = 0;
    static constexpr int offS  = (offQ + kBM * kQP * 2 + 15) & ~15;
    static constexpr int offP  = (offS + kBM * kSP * 4 + 15) & ~15;
    static constexpr int offKV = (offP + kBM * kPP * 2 + 15) & ~15;
    static constexpr int offM  = (offKV + kBN * kVP * 2 + 15) & ~15;   /* + m[64] l[64] */
    static constexpr int offRB = offM + 2 * kBM * 4;
    static constexpr int offCB = offRB + kBN * 8;
    static constexpr int total = offCB + kBN + 16;
};

template <int D, int KVDT>
__global__ __launch_bounds__(kThreads) void attn_paged_wmma_kernel(R3dAttnParams P) {
    constexpr int DSH = (D == 128) ? 7 : 6, DM1 = D - 1;
    typedef Carve<D> C;
    __shared__ unsigned char raw[C::total];

    bf16*   Qsp = reinterpret_cast<bf16*>(raw + C::offQ);
    float*  Ssp = reinterpret_cast<float*>(raw + C::offS);
    bf16*   Psp = reinterpret_cast<bf16*>(raw + C::offP);
    bf16*   KVp = reinterpret_cast<bf16*>(raw + C::offKV);
    float*  MSP = reinterpret_cast<float*>(raw + C::offM);
    float*  LSP = MSP + kBM;
    int64_t* RB = reinterpret_cast<int64_t*>(raw + C::offRB);
    unsigned char* CB = raw + C::offCB;

    const int h = blockIdx.x, s = blockIdx.y, qt = blockIdx.z;
    const int tid = threadIdx.x, w = tid >> 5;

    /* ---- per-block bookkeeping. blockIdx carries (head, seq, query tile) directly, so
     * none of v1's linear seq_of scan runs on device; every thread reads the two or three
     * uniform scalars itself -- cheaper than a thread-0 broadcast at 128 threads. ---- */
    int64_t start, qlen;
    if (P.cus) {
        start = (int64_t)ldi_w(P.cus_dt, P.cus, s);
        qlen  = (int64_t)ldi_w(P.cus_dt, P.cus, s + 1) - start;
    } else {
        qlen  = P.q_len_uniform;
        start = (int64_t)s * qlen;
    }
    const int64_t i0 = (int64_t)qt * kBM;
    if (i0 >= qlen) return;
    const int64_t tok_base = start + i0;
    const int64_t valid    = min<int64_t>(qlen - i0, kBM);
    const int64_t ctx      = (int64_t)ldi_w(P.su_dt, P.su, s);
    const int64_t pos_base = ctx - qlen + i0;
    const int64_t kh       = h / P.gqa;
    const float   kds      = P.kdes ? ldt_w(P.kdes, s * P.kv_heads + kh) : 1.0f;
    const float   vds      = P.vdes ? ldt_w(P.vdes, s * P.kv_heads + kh) : 1.0f;
    const float   scale2   = P.scale * kds * kLog2e;
    const int64_t jend     = min<int64_t>(ctx, P.jcap);

    /* ref's key_at visibility rules, minus the addressing (key_at's mask half). */
    const int causal = P.causal; const int64_t window = P.window;
    auto visible = [&](int64_t rowpos, int64_t j) -> bool {
        if (causal && j > rowpos) return false;
        if (window > 0) {
            const int64_t d = rowpos - j;
            if (causal ? (d >= window) : (d >= window || d <= -window)) return false;
        }
        return true;
    };

    /* Tile range that any row can see (skip the dead head of a window, stop past the
     * last position any row may attend to). */
    const int64_t pos_last = pos_base + valid - 1;
    int64_t jmax = jend;
    if (causal && pos_last + 1 < jmax) jmax = pos_last + 1;
    if (!causal && window > 0 && pos_last + window < jmax) jmax = pos_last + window;
    int64_t jmin = 0;
    if (window > 0) jmin = max<int64_t>(0, pos_base - window + 1);
    const int t0 = (int)(jmin / kBN);
    const int t1 = (int)((jmax + kBN - 1) / kBN);

    /* ---- stage Q once: rows tok_base..+63 (invalid rows zeroed; they mask everywhere
     * and never store), columns 0..head_dim-1. ---- */
    const int64_t qbase = tok_base * P.q_rs + (int64_t)h * P.q_hs;
    for (int e = tid; e < kBM * D; e += kThreads) {
        const int r = e >> DSH, d = e & DM1;
        Qsp[r * C::kQP + d] = (r < valid)
            ? bf16(((const uint16_t*)P.q)[qbase + (int64_t)r * P.q_rs + (int64_t)d * P.q_ds])
            : bf16((uint16_t)0);
    }
    if (tid < kBM) { MSP[tid] = -INFINITY; LSP[tid] = 0.0f; }
    __syncthreads();

    roCW::fragment<roCW::matrix_a, 16, 16, 16, bf16, roCW::row_major> fa;
    roCW::fragment<roCW::matrix_b, 16, 16, 16, bf16, roCW::col_major> fb0, fb1;
    roCW::fragment<roCW::accumulator, 16, 16, 16, float> a0, a1;

    /* ================= pass one: per-row running max m and exp-sum l ================= */
    for (int t = t0; t < t1; ++t) {
        const int64_t jb = (int64_t)t * kBN;

        /* key addressing for the 32 columns (one thread per key, then broadcast) */
        if (tid < kBN) {
            const int64_t j = jb + tid;
            int64_t rb = -1;
            if (j < jend) {
                const int64_t blk = (int64_t)ldi_w(P.bt_dt, P.bt,
                                                   s * P.bt_rs + (j / P.block_size) * P.bt_cs);
                if (blk >= 0)
                    rb = blk * P.kv_sb + kh * P.kv_sh + (j % P.block_size) * P.kv_so;
            }
            RB[tid] = rb; CB[tid] = (rb < 0) ? 1 : 0;
        }
        __syncthreads();

        for (int e = tid; e < kBN * D; e += kThreads) {
            const int r = e >> DSH, d = e & DM1;
            uint16_t u = 0;
            if (RB[r] >= 0) {
                const int64_t a = RB[r] + (int64_t)d * P.kv_sd;
                u = (KVDT == RAD_BF16) ? ((const uint16_t*)P.kv)[a]
                                       : fp8_to_bf16_w(((const uint8_t*)P.kv)[a]);
            }
            KVp[r * C::kVP + d] = bf16(u);
        }
        __syncthreads();

        /* S = Q.K^T: matrix_a row-major at &Qsp[16w][k0]; the [k][n] B tile is K^T, and
         * col-major at &KVp[n0][k0] means element (k,n) = ptr[n*VP + k] = KVp[n][k]. */
        roCW::fill_fragment(a0, 0.0f);
        roCW::fill_fragment(a1, 0.0f);
        for (int k0 = 0; k0 < D; k0 += 16) {
            roCW::load_matrix_sync(fa,  Qsp + (w * 16) * C::kQP + k0, C::kQP, roCW::mem_row_major);
            roCW::load_matrix_sync(fb0, KVp + k0, C::kVP);
            roCW::load_matrix_sync(fb1, KVp + 16 * C::kVP + k0, C::kVP);
            roCW::mma_sync(a0, fa, fb0, a0);
            roCW::mma_sync(a1, fa, fb1, a1);
        }
        roCW::store_matrix_sync(Ssp + (w * 16) * C::kSP,      a0, C::kSP, roCW::mem_row_major);
        roCW::store_matrix_sync(Ssp + (w * 16) * C::kSP + 16, a1, C::kSP, roCW::mem_row_major);
        __syncthreads();

        /* one thread per query row: visible max, then the online update of (m, l) */
        if (tid < kBM && tid < valid) {
            const int64_t rowpos = pos_base + tid;
            const float* srow = Ssp + tid * C::kSP;
            float mx = MSP[tid];
            for (int jj = 0; jj < kBN; ++jj) {
                if (CB[jj] || !visible(rowpos, jb + jj)) continue;
                const float v = srow[jj] * scale2;
                if (v > mx) mx = v;
            }
            if (mx != -INFINITY) {
                if (mx > MSP[tid]) {
                    if (MSP[tid] != -INFINITY) LSP[tid] *= exp2f(MSP[tid] - mx);
                    MSP[tid] = mx;
                }
                float sum = 0.0f;
                for (int jj = 0; jj < kBN; ++jj) {
                    if (CB[jj] || !visible(rowpos, jb + jj)) continue;
                    sum += exp2f(srow[jj] * scale2 - mx);
                }
                LSP[tid] += sum;
            }
        }
        __syncthreads();
    }

    /* ===================== pass two: O = P.V, P at the final m ====================== */
    roCW::fragment<roCW::accumulator, 16, 16, 16, float> oa[D / 16];
    #pragma unroll
    for (int i = 0; i < D / 16; ++i) roCW::fill_fragment(oa[i], 0.0f);

    for (int t = t0; t < t1; ++t) {
        const int64_t jb = (int64_t)t * kBN;

        if (tid < kBN) {                       /* key addressing, as pass one */
            const int64_t j = jb + tid;
            int64_t rb = -1;
            if (j < jend) {
                const int64_t blk = (int64_t)ldi_w(P.bt_dt, P.bt,
                                                   s * P.bt_rs + (j / P.block_size) * P.bt_cs);
                if (blk >= 0)
                    rb = blk * P.kv_sb + kh * P.kv_sh + (j % P.block_size) * P.kv_so;
            }
            RB[tid] = rb; CB[tid] = (rb < 0) ? 1 : 0;
        }
        __syncthreads();
        for (int e = tid; e < kBN * D; e += kThreads) {
            const int r = e >> DSH, d = e & DM1;
            uint16_t u = 0;
            if (RB[r] >= 0) {
                const int64_t a = RB[r] + (int64_t)d * P.kv_sd;
                u = (KVDT == RAD_BF16) ? ((const uint16_t*)P.kv)[a]
                                       : fp8_to_bf16_w(((const uint8_t*)P.kv)[a]);
            }
            KVp[r * C::kVP + d] = bf16(u);
        }
        __syncthreads();

        roCW::fill_fragment(a0, 0.0f);         /* S again: same fragments, same order, */
        roCW::fill_fragment(a1, 0.0f);         /* so P here matches pass one's (m, l)  */
        for (int k0 = 0; k0 < D; k0 += 16) {
            roCW::load_matrix_sync(fa,  Qsp + (w * 16) * C::kQP + k0, C::kQP, roCW::mem_row_major);
            roCW::load_matrix_sync(fb0, KVp + k0, C::kVP);
            roCW::load_matrix_sync(fb1, KVp + 16 * C::kVP + k0, C::kVP);
            roCW::mma_sync(a0, fa, fb0, a0);
            roCW::mma_sync(a1, fa, fb1, a1);
        }
        roCW::store_matrix_sync(Ssp + (w * 16) * C::kSP,      a0, C::kSP, roCW::mem_row_major);
        roCW::store_matrix_sync(Ssp + (w * 16) * C::kSP + 16, a1, C::kSP, roCW::mem_row_major);
        __syncthreads();

        for (int e = tid; e < kBM * kBN; e += kThreads) {   /* S -> P, all four waves */
            const int r = e >> 5, jj = e & 31;
            float pv = 0.0f;
            if (r < valid && !CB[jj] && visible(pos_base + r, jb + jj)) {
                const float m = MSP[r];
                if (m != -INFINITY) pv = exp2f(Ssp[r * C::kSP + jj] * scale2 - m);
            }
            Psp[r * C::kPP + jj] = bf16(f32_bf16_w(pv));
        }
        __syncthreads();                       /* this tile's S is consumed: KV is free */

        for (int e = tid; e < kBN * D; e += kThreads) {     /* V tile, aliasing K */
            const int r = e >> DSH, d = e & DM1;
            uint16_t u = 0;
            if (RB[r] >= 0) {
                const int64_t a = RB[r] + ((int64_t)D + d) * P.kv_sd;
                u = (KVDT == RAD_BF16) ? ((const uint16_t*)P.kv)[a]
                                       : fp8_to_bf16_w(((const uint8_t*)P.kv)[a]);
            }
            KVp[r * C::kVP + d] = bf16(u);
        }
        __syncthreads();

        /* O += P.V: A = P row-major [16w][jk]; the V tile IS [key][d] = row-major B. */
        for (int jk = 0; jk < kBN / 16; ++jk) {
            roCW::fragment<roCW::matrix_a, 16, 16, 16, bf16, roCW::row_major> fap;
            roCW::load_matrix_sync(fap, Psp + (w * 16) * C::kPP + jk * 16, C::kPP,
                                   roCW::mem_row_major);
            #pragma unroll
            for (int d0 = 0; d0 < D / 16; ++d0) {
                roCW::fragment<roCW::matrix_b, 16, 16, 16, bf16, roCW::row_major> fbv;
                roCW::load_matrix_sync(fbv, KVp + (jk * 16) * C::kVP + d0 * 16, C::kVP,
                                       roCW::mem_row_major);
                roCW::mma_sync(oa[d0], fap, fbv, oa[d0]);
            }
        }
        __syncthreads();
    }

    /* ===================== out = (O / l) * vds, in 32-row halves ===================== */
    float* Osp = reinterpret_cast<float*>(raw + C::offQ);   /* aliases Q+S, both dead */
    for (int half = 0; half < 2; ++half) {
        __syncthreads();
        /* wave owns rows [16w,16w+16): waves 0/1 fill half 0, waves 2/3 fill half 1.
         * PREDICATED, never `continue`d: every thread must reach both syncthreads. */
        if ((half == 0) ? (w < 2) : (w >= 2)) {
            const int wr = w - 2 * half;        /* relative tile-row: 0 or 1 */
            #pragma unroll
            for (int d0 = 0; d0 < D / 16; ++d0)
                roCW::store_matrix_sync(Osp + (wr * 16) * C::kOP + d0 * 16, oa[d0], C::kOP,
                                        roCW::mem_row_major);
        }
        __syncthreads();
        for (int e = tid; e < 32 * D; e += kThreads) {
            const int r = e >> DSH, d = e & DM1;
            const int row = half * 32 + r;
            if (row >= valid) continue;
            const float ll = LSP[row];
            const float o = Osp[r * C::kOP + d] * (ll > 0.0f ? 1.0f / ll : 0.0f) * vds;
            const int64_t ob = (tok_base + row) * P.o_rs + (int64_t)h * P.o_hs
                             + (int64_t)d * P.o_ds;
            if (P.out_f32) ((float*)P.out)[ob] = o;
            else           ((uint16_t*)P.out)[ob] = f32_bf16_w(o);
        }
    }
}

} /* anonymous namespace */

#else  /* generic + non-RDNA3: kernel exists as a no-op so the symbol links; the
        * launcher's gcnArchName gate below never dispatches it on such a chip. */

namespace {
template <int D, int KVDT>
__global__ __launch_bounds__(kThreads) void attn_paged_wmma_kernel(R3dAttnParams) {}
} /* anonymous namespace */

#endif

int r3d_attn_paged_wmma_launch(const R3dAttnParams& P, hipStream_t stream) {
    static int allowed = -1;
    if (allowed < 0) {
        hipDeviceProp_t d;
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&d, dev) != hipSuccess)
            allowed = 0;
        else
            allowed = (strstr(d.gcnArchName, "gfx1100") || strstr(d.gcnArchName, "gfx1101") ||
                       strstr(d.gcnArchName, "gfx1102") || strstr(d.gcnArchName, "gfx1103")) ? 1 : 0;
        if (getenv("R3D_NO_WMMA")) allowed = 0;      /* one switch for every WMMA row */
    }
    if (!allowed) return RAD_E_UNSUPPORTED;
    if (P.qdt != RAD_BF16) return RAD_E_UNSUPPORTED;
    if (P.head_dim != 64 && P.head_dim != 128) return RAD_E_UNSUPPORTED;

    const unsigned tiles = (unsigned)((P.rows + kBM - 1) / kBM);
    if (P.n_seq <= 0 || P.n_seq > 65535 || tiles == 0 || tiles > 65535) return RAD_E_UNSUPPORTED;
    dim3 grid((unsigned)P.q_heads, (unsigned)P.n_seq, tiles);
    dim3 block(kThreads);

    const int kvbf = (P.kvdt == RAD_BF16);
    if (!kvbf && P.kvdt != RAD_F8E4M3) return RAD_E_UNSUPPORTED;
    #define R3D_LAUNCH(D, KV) hipLaunchKernelGGL((attn_paged_wmma_kernel<D, KV>), grid, block, \
                                                 0, stream, P)
    const int d128 = (P.head_dim == 128);
    if      (d128 &&  kvbf) R3D_LAUNCH(128, RAD_BF16);
    else if (d128 && !kvbf) R3D_LAUNCH(128, RAD_F8E4M3);
    else if (!d128 &&  kvbf) R3D_LAUNCH(64, RAD_BF16);
    else                     R3D_LAUNCH(64, RAD_F8E4M3);
    #undef R3D_LAUNCH
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
