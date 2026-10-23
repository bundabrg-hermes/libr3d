/* r3d_gemm_mxfp4.hip -- gemm_nt_q at the `mxfp4a8` encoding on RDNA3.

 * Y[M,N] bf16 = A[M,K] fp8_e4m3 @ B[N,K] e2m1^T, block-scaled:
 *   a_scale  f32 [M, ceil(K/128)]   per (row, 128 of K), from quant_act_fp8
 *   b_scale  u8/E8M0 [N, ceil(K/32)] per (row, 32 of K), 2^(byte-127), 0xFF = NaN
 *   a_sum absent (the entry refuses it); b_ref accepted and IGNORED -- like both host
 *   libraries: it is "derived from b_scale and does not change the product" (docs/OPS.md),
 *   and this tile derives its own per-row reference at block entry.
 *
 * RDNA3 has no fp8/fp4 tensor path, so this is a boundary-conversion design on the m64
 * bf16 tile: both decodes are EXACT in bf16 (e4m3 mantissa <= 3 bits, e2m1 <= 2), and the
 * weight scales -- powers of two -- never touch bf16 rounding either:
 *
 *   ref_gemm.h quant_dot computes, per 32-of-K group g: sub_g = sum_{k in g} a_k*b_k
 *   (ascending fp32), then  acc += (sub_g * sa[m, g/4]) * 2^(x_{n,g}-127).
 *
 *   Here: B fragments are staged as e2m1_bits + ((x - eref[n]) << 7) -- bf16's biased
 *   exponent makes that add IS the multiply by 2^(x-eref[n]), exact while normal; a group
 *   whose folded scale leaves bf16's normal range contributes ~2^-126 relative to the
 *   row, which no tolerance can see. eref[n] is the row's max non-NaN exponent (the b_ref
 *   role, self-derived). WMMA then accumulates raw products for exactly one 128-of-K
 *   activation group (8 slices of kKB=16; 32 divides 128), the tile flushes through LDS,
 *   multiplies by sa[m, grp] in f32 -- one rounding per 128 k where the oracle rounds per
 *   32, inside the same rounding-order class the mma's opaque k-tree already belongs to
 *   (and (sub*2^d)*sa == (sub*sa)*2^d: one rounding of the same mantissa) -- and the store
 *   multiplies once more by 2^(eref[n]-127). Power of two, exact.
 *
 *   NaN scale bytes (0xFF) stage their whole 32-column group as bf16 NaN: the oracle's
 *   (sub*sa)*NaN poisons the entire output -- even an all-zero-code group yields 0*NaN --
 *   and OPS.md's "treat as empty" is a permission for ZERO activation scales, not for
 *   NaN weight scales. Silencing it here would be a kbench FAIL dressed as robustness.
 *   Out-of-range padding (m>=M, n>=N, k>=K) stages plain zeros instead -- the oracle
 *   never sees those columns at all.
 *
 * The flush accumulator lives in LDS (Acc) because this ROCm's rocwmma has no coordinate
 * fragment (no rocm_t): a register-side per-row multiply would have to hardcode v_wmma's
 * lane layout, and the LDS round trip is 16 KB every 12 KB of payload -- this GEMM is
 * weight-bandwidth-bound (M=1 decode reads 64 MB of nibbles per call at N=5120 K=25600),
 * so LDS traffic is not the wall. The ~43 KB carve makes it 1 block/CU; at 80..2560
 * blocks over 96 CUs that costs nothing a barrier could exploit.
 *
 * -ffp-contract=off and the wave32 pin live in CMakeLists (plugin-wide). Chip gate and
 * the both-schedules-in-one-TU rule are the r3d_gemm_bf16_m64.hip pattern: DBUF=1 (the
 * default) pipelines the next slice's global loads into registers while WMMA runs, A
 * double-buffered, B single (block 06's measured rule); R3D_GEMM_NO_DBUF=1 takes the
 * serialized arm; R3D_NO_MXFP4=1 refuses the row entirely so the engine answers from the
 * host reference -- the serve A/B kill switch, no remount needed.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "r3d_dump.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rad_abi.h"
#include "rad_plugin.h"   /* rad_arg_in / rad_tensor_numel (host-side entry helpers) */

namespace {

/* ref_gemm.h grid_block, verbatim: the block size a scale grid of `blocks` columns
 * implies over K. The entry accepts ONLY the grid the vocabulary spells for mxfp4a8
 * (b: 32, a: 128), deriving it from the operands rather than trusting the `group`
 * parameter -- the same rule the oracle's make_scales follows. */
int64_t m4_grid_block(int64_t ext, int64_t blocks) {
    if (ext <= 0) return 1;
    if (blocks <= 1) return ext;
    const int64_t lo = (ext + blocks - 1) / blocks;
    int64_t b = 1;
    while (b < lo) b <<= 1;
    return (ext + b - 1) / b == blocks ? b : lo;
}

/* ref_gemm.h row_stride: fold trailing dims until they span `cols`; the next axis out is
 * the row. Same fold the oracle reads operands with. */
int64_t m4_row_stride(const RadTensor* t, int64_t cols) {
    if (!t || t->rank < 2) return cols;
    int64_t span = 1;
    for (int i = (int)t->rank - 1; i >= 0; --i) {
        span *= t->shape[i];
        if (span == cols) return (i > 0) ? t->stride[i - 1] : cols;
        if (span > cols) break;
    }
    return t->stride[t->rank - 2];
}

} /* anonymous namespace */

constexpr int kMB = 64, kNB = 64, kKB = 16;   /* same tile as the bf16 m64 */
constexpr int kLdsPad = 4;
constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kItems = (kMB * kKB) / kThreads;      /* 4 per thread per slice */

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)

#include <rocwmma/rocwmma.hpp>

namespace {

using bf16 = rocwmma::bfloat16_t;

__device__ __forceinline__ uint16_t f32_to_bf16_m4(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* rad_plugin.h rad_fp8e4m3_to_f32, ported verbatim -- it is host-only linkage under
 * HIP's clang, and this is the oracle's own decode; re-implementing its semantics
 * separately would be a second place for the spec to drift. e4m3fn: 1-4-3, bias 7,
 * no infinity, S1111111 the only NaN, max 448. */
__device__ __forceinline__ float fp8e4m3_m4(uint8_t b) {
    uint32_t s = (uint32_t)(b >> 7) << 31;
    uint32_t e = (uint32_t)(b >> 3) & 0xfu;
    uint32_t m = (uint32_t)b & 0x7u;
    uint32_t u;
    if (e == 0u) {
        if (m == 0u) {
            u = s;
        } else {
            int sh = 0;
            while (!(m & 0x8u)) { m <<= 1; ++sh; }
            m &= 0x7u;
            u = s | ((uint32_t)(127 - 6 - sh) << 23) | (m << 20);
        }
    } else if (e == 0xfu && m == 0x7u) {
        u = s | 0x7fc00000u;                                  /* the single NaN encoding */
    } else {
        u = s | ((e - 7u + 127u) << 23) | (m << 20);
    }
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* rad_plugin.h rad_fp8e4m3_to_f32, loop-free form (the fp8 TU's gemv decoder ported):
 * the subnormal branch is m * 2^-9 exactly. Bit-equivalent to fp8e4m3_m4 above; this
 * sits in the bandwidth row's staging path, where a per-byte divergent while-loop has
 * no business. */
__device__ __forceinline__ float fp8e4m3_gemv(uint8_t b) {
    const uint32_t s = (uint32_t)(b >> 7) << 31;
    const uint32_t e = (uint32_t)(b >> 3) & 0xfu, m = (uint32_t)b & 0x7u;
    uint32_t u;
    if (e == 0u) {
        if (m == 0u) return __uint_as_float(s);
        const float f = (float)m / 512.0f;                                /* m * 2^-9 */
        u = s | (__float_as_uint(f) & 0x7fffffffu);
    } else if (e == 0xfu && m == 0x7u) {
        u = s | 0x7fc00000u;                                              /* the NaN */
    } else {
        u = s | ((e - 7u + 127u) << 23) | (m << 20);
    }
    return __uint_as_float(u);
}

/* e2m1 as bf16 bit patterns: 0, .5, 1, 1.5, 2, 3, 4, 6 (all exact) with the sign. */
__device__ const uint16_t kE2M1[16] = {
    0x0000u, 0x3F00u, 0x3F80u, 0x3FC0u, 0x4000u, 0x4040u, 0x4080u, 0x40C0u,
    0x8000u, 0xBF00u, 0xBF80u, 0xBFC0u, 0xC000u, 0xC040u, 0xC080u, 0xC0C0u,
};

/* One 64x64 output tile per workgroup, K in 16-wide slices, flushed per activation
 * group of `ag` (128) through LDS. B slices are staged DECODED and FOLDED; A slices
 * decoded only (the scale that depends on m rides the flush, not the fragment). */
template <bool DBUF>
__global__ __launch_bounds__(kThreads) void gemm_nt_mxfp4_m64_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, uint16_t* __restrict__ Y,
    const float* __restrict__ AS, const uint8_t* __restrict__ BS,
    int M, int N, int K, int as_cols, int bs_cols,
    int64_t a_pitch, int64_t b_pitch,        /* B's pitch is BYTES (nibbles: K/2 of them) */
    int64_t y_pitch, int64_t as_pitch, int64_t bs_pitch,
    int ag, int bg) {                       /* scale grids, entry-validated: 128 and 32 */

    __shared__ uint16_t As[DBUF ? 2 : 1][kMB][kKB + kLdsPad];
    __shared__ uint16_t Bs[kNB][kKB + kLdsPad];
    __shared__ float    Acc[kMB][kNB + kLdsPad];   /* persistent, per-group-flushed */
    __shared__ float    Ps[kMB][kNB + kLdsPad];    /* per-flush WMMA partial staging */
    __shared__ uint8_t  Eref[kNB];                 /* per-row reference exponent */
    __shared__ uint16_t Lut4m3[256];               /* e4m3 byte -> exact bf16 bits */

    const int m0 = blockIdx.y * kMB;
    const int n0 = blockIdx.x * kNB;
    const int w = threadIdx.x / kWave;
    const int tw = w >> 1;
    const int tc = (w & 1) * 2;

    /* Prologue: LUT + per-row reference exponents + zeroed accumulator. The LUT uses
     * the ABI header's decoder -- the same conversion the oracle's loader performs,
     * so fp8 decode cannot drift from the spec a second time. */
    if (threadIdx.x < 256)
        Lut4m3[threadIdx.x] = f32_to_bf16_m4(fp8e4m3_m4((uint8_t)threadIdx.x));
    if (threadIdx.x < kNB) {
        const int n = n0 + threadIdx.x;
        uint8_t mx = 0;
        if (n < N)
            for (int j = 0; j < bs_cols; ++j) {
                const uint8_t x = BS[(int64_t)n * bs_pitch + j];
                if (x != 0xffu && x > mx) mx = x;
            }
        Eref[threadIdx.x] = mx;
    }
    for (int i = threadIdx.x; i < kMB * (kNB + kLdsPad); i += kThreads)
        Acc[i / (kNB + kLdsPad)][i % (kNB + kLdsPad)] = 0.0f;
    __syncthreads();                        /* LUT, Eref, Acc visible */

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bf16, rocwmma::row_major> fa;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bf16, rocwmma::col_major> fb0, fb1;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fc0, fc1;
    rocwmma::fill_fragment(fc0, 0.0f);
    rocwmma::fill_fragment(fc1, 0.0f);

    const int c_col = threadIdx.x % kKB;
    const int r_row = threadIdx.x / kKB;

    /* Per thread per slice: 4 items at (row = r_row + 16j, col = c_col) -- the bf16 m64
     * staging map, so A and B share one walk pattern. A items hold decoded bf16 bits;
     * B items hold the raw nibble BYTE, its slice's scale byte, and a liveness bit that
     * keeps padding (zero) distinguishable from a NaN scale (NaN). */
    uint16_t areg[kItems];
    uint8_t  bcode[kItems], bexp[kItems];
    bool     blive[kItems];

    auto prefetch = [&](int k0) {
        #pragma unroll
        for (int j = 0; j < kItems; ++j) {
            const int i = threadIdx.x + j * kThreads;
            const int ra = i / kKB, ka = k0 + i % kKB;   /* ka inside a 16-wide slice */
            const int ma = m0 + ra;
            areg[j] = (ma < M && ka < K)
                ? Lut4m3[A[(int64_t)ma * a_pitch + ka]] : (uint16_t)0;
            const int nb = n0 + ra;
            blive[j] = (nb < N && ka < K);
            if (blive[j]) {
                bcode[j] = B[(int64_t)nb * b_pitch + (ka >> 1)];
                /* One scale byte per (row, 32-group): kKB=16 divides bg=32, so a slice
                 * never straddles a scale group. */
                bexp[j] = BS[(int64_t)nb * bs_pitch + ka / bg];
            } else { bcode[j] = 0; bexp[j] = 0; }
        }
    };
    auto stage_a = [&](int ab) {
        #pragma unroll
        for (int j = 0; j < kItems; ++j)
            As[ab][r_row + j * (kThreads / kKB)][c_col] = areg[j];
    };
    auto stage_b = [&]() {
        #pragma unroll
        for (int j = 0; j < kItems; ++j) {
            const int r = r_row + j * (kThreads / kKB);
            uint16_t bits;
            if (!blive[j])                          bits = 0u;        /* padding, not NaN */
            else if (bexp[j] == 0xffu)              bits = 0x7FC0u;   /* NaN poisons group */
            else {
                const uint8_t nib = (c_col & 1) ? (uint8_t)(bcode[j] >> 4)
                                                : (uint8_t)(bcode[j] & 0xfu);
                if ((nib & 7u) == 0u) bits = kE2M1[nib];              /* +-0: no fold */
                else {
                    /* Exponent add in bf16 bit space == multiply by 2^d, exact while
                     * normal; d <= 0 (eref is the row max), so underflow-only: flushing
                     * to +-0 below bf16's subnormal floor drops ~2^-126 relative. */
                    const int d = (int)bexp[j] - (int)Eref[r];
                    const int nb = (int)kE2M1[nib] + (d << 7);
                    bits = (uint16_t)((nb < 0x0080) ? (int)(nib & 8u ? 0x8000u : 0u) : nb);
                }
            }
            Bs[r][c_col] = bits;
        }
    };

    auto compute = [&](int ab) {
        const bf16* ap  = reinterpret_cast<const bf16*>(&As[ab][tw * 16][0]);
        const bf16* bp0 = reinterpret_cast<const bf16*>(&Bs[tc * 16][0]);
        const bf16* bp1 = reinterpret_cast<const bf16*>(&Bs[(tc + 1) * 16][0]);
        rocwmma::load_matrix_sync(fa,  ap,  kKB + kLdsPad, rocwmma::mem_row_major);
        rocwmma::load_matrix_sync(fb0, bp0, kKB + kLdsPad);
        rocwmma::load_matrix_sync(fb1, bp1, kKB + kLdsPad);
        rocwmma::mma_sync(fc0, fa, fb0, fc0);
        rocwmma::mma_sync(fc1, fa, fb1, fc1);
    };

    /* One activation group's partial: stage the WMMA accumulators to LDS, fold the
     * activation scale in per row, and bank into Acc. Called at uniform ks, so barrier
     * counts never diverge across the block. */
    auto flush = [&](int grp) {
        rocwmma::store_matrix_sync(&Ps[tw * 16][tc * 16],       fc0, kNB + kLdsPad, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(&Ps[tw * 16][(tc + 1) * 16], fc1, kNB + kLdsPad, rocwmma::mem_row_major);
        rocwmma::fill_fragment(fc0, 0.0f);
        rocwmma::fill_fragment(fc1, 0.0f);
        __syncthreads();                       /* Ps fully written by every wave */
        for (int i = threadIdx.x; i < kMB * kNB; i += kThreads) {
            const int r = i / kNB, c = i % kNB;
            const int m = m0 + r;
            if (m < M && grp < as_cols)
                Acc[r][c] += Ps[r][c] * AS[(int64_t)m * as_pitch + grp];
        }
    };

    const int nks = (K + kKB - 1) / kKB;
    const int kFlush = ag / kKB;               /* slices per activation group */

    if constexpr (DBUF) {
        prefetch(0);
        stage_a(0); stage_b();
        __syncthreads();                       /* prologue slices visible */
        int held = 0, grp = 0;
        for (int ks = 0; ks < nks; ++ks) {
            if (ks + 1 < nks) prefetch((ks + 1) * kKB);
            if (ks + 1 < nks) stage_a((ks + 1) & 1);   /* other A buffer: overlaps WMMA */

            compute(ks & 1);
            ++held;

            __syncthreads();                   /* this slice's As[ab]/Bs reads done */
            if (held == kFlush || ks == nks - 1) { flush(grp++); held = 0; }

            if (ks + 1 < nks) stage_b();       /* Bs single-buffered: after the barrier */
            __syncthreads();                   /* next slices visible */
        }
    } else {
        int held = 0, grp = 0;
        for (int ks = 0; ks < nks; ++ks) {
            prefetch(ks * kKB);
            stage_a(0); stage_b();
            __syncthreads();                   /* staged slice visible */

            compute(0);
            ++held;

            __syncthreads();                   /* reads done; next stage may overwrite */
            if (held == kFlush || ks == nks - 1) { flush(grp++); held = 0; }
        }
        __syncthreads();                       /* flush's Acc writes complete */
    }
    __syncthreads();

    /* Store: the row's reference exponent rides out once per output -- ldexpf by an
     * integer power of two is exact (and NaN/0 pass through untouched). */
    for (int i = threadIdx.x; i < kMB * kNB; i += kThreads) {
        const int r = i / kNB, c = i % kNB;
        const int m = m0 + r, n = n0 + c;
        if (m < M && n < N)
            Y[(int64_t)m * y_pitch + n] = f32_to_bf16_m4(ldexpf(Acc[r][c], (int)Eref[c] - 127));
    }
}

/* ---- the decode-band width row (issue #5) -------------------------------------------
 *
 * The m64 tile at M = 1..8 stages every nibble row through a 64-row slab it uses M rows
 * of: ~28 GB/s effective on the flagship serve's decode. This row reads every weight byte
 * exactly once instead -- warp per weight row, eight codes per lane (one aligned dword:
 * the fp8 GEMV's uint4-per-lane walk nibble-halved, 4 bytes carrying 8 e2m1 codes),
 * the MT activation rows
 * decoded through LDS per 256-of-K span.
 *
 * The fold keeps the ORACLE's per-32-of-K flush, which the tile gave up: a 256 span is
 * exactly eight 32-of-K scale groups and each 4-lane quad owns one, so the k chain folds
 * inside the quad (butterfly offsets 1,2), the quad flushes (sub * sa) * 2^(x-127) in
 * quant_dot's two-multiply order, and the eight flush terms fold across quads (offsets
 * 4,8,16) into acc. That tree-reassociates only the eight group terms; the tile
 * reassociates the whole 32-term chain AND defers sa by four groups. No eref here: the
 * f32 path needs no bf16 exponent fold, so the store is the plain bf16 narrow of the
 * oracle's acc.
 *
 * The E8M0 decode mirrors ref_common.h e8m0_to_f32 exactly: 0xFF is NaN and poisons acc
 * through (sub*sa)*NaN like the oracle (the fp8 GEMV's clamped-scale-row trick does not
 * apply -- mxfp4's b_scale is per weight row, so dead quads are folded in as exact zeros
 * by predicate, never by a clamped scale byte); 0x00 is the subnormal 2^-127 that
 * ldexpf gives the oracle and the bare bit trick gives zero; every other byte is the
 * exact biased-exponent bit pattern.
 *
 * Ragged N clamps dead warps' load row (never an early return: every warp reaches the
 * block's barriers). The K % 128 == 0 row gate means a tail span is exactly 128 wide --
 * quads 4..7 predicate off entirely (no OOB scale byte, no OOB row read), so a K that is
 * a multiple of 128 but not of 256 costs one half-width span, not a row refusal.
 */
/* PF (issue #7): window burst of W spans' weight-code dwords + scale bytes, issued
 * before any consume (same loads, same lane mapping — schedule-only, bit-exact).
 * MEASURED: batching loads makes mxfp4 WORSE (W=4 with activations in the window:
 * 67 vs 142 GB/s at 4096^2, 121 vs 346 at 16K; even the 2-reg b4+sc window at
 * MT>=2 lost 15%). The #5 loop is already self-pipelined by the scheduler and the
 * family is not MLP-starved — the profile prescription is falsified for this GEMV.
 * Default dispatch is therefore W=1 (the #5 arm verbatim); R3D_FP4_GEMV_PF=1 opts
 * into W=4 for re-measurement. fp8 is the opposite story and ships W=8. */
template <int MT, int W>
__global__ __launch_bounds__(kThreads) void gemm_nt_mxfp4_gemv_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, uint16_t* __restrict__ Y,
    const float* __restrict__ AS, const uint8_t* __restrict__ BS,
    int M, int N, int K,
    int64_t a_pitch, int64_t b_pitch,        /* BYTES (nibbles: K/2 of them) */
    int64_t y_pitch, int64_t as_pitch, int64_t bs_pitch)
{
    __shared__ __align__(16) float xs[MT][256];
    __shared__ float L16[16];                 /* e2m1 nibble -> exact f32 (kE2M1 << 16) */

    if (threadIdx.x < 16)
        L16[threadIdx.x] = __uint_as_float((uint32_t)kE2M1[threadIdx.x] << 16);

    const int w    = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int r    = blockIdx.x * (kThreads / 32) + w;
    const int rr   = r < N ? r : N - 1;               /* dead warps load a live row */
    const int live = M < MT ? M : MT;

    const uint8_t* crow = B + (int64_t)rr * b_pitch;
    const uint8_t* brow = BS + (int64_t)rr * bs_pitch;   /* b_scale is per weight row */

    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;
    __syncthreads();                            /* LUT visible (and xs, where it exists) */

    const int n256 = K >> 8;                      /* K % 128 == 0 is the row gate */
    const bool partial = (K & 255) != 0;          /* one trailing 128-wide span */

    if constexpr (MT == 1) {
        /* M == 1 -- the decode step. No xs: 16 codes per lane (uint2 of nibbles plus
         * the matching uint4 of activation bytes) ride registers side by side, so the
         * span loop carries NO barriers at all, at the fp8 GEMV's load widths.
         * A 512-span is sixteen 32-of-K groups: lane PAIR p (lanes 2p, 2p+1) folds to
         * own group it*16+p exactly -- pair butterfly (offset 1), the pair flushes
         * (sub * sa) * 2^(x-127), and the sixteen terms fold warp-wide (offsets 2..16).
         * Odd A bases take a scalar byte walk, never an issue-time refusal (#4 rule). */
        float a0 = 0.0f;
        const int nspan = K >> 9;
        const int tail  = K & 511;                   /* K % 128 == 0 gate: 128 or 384 */
        const int ntot  = nspan + (tail ? 1 : 0);

        auto load_act = [&](int it, uint4& av) {
            const int kb  = it << 9;
            const int rem = (it < nspan) ? 512 : tail;
            const bool act = (lane << 4) < rem;
            av = make_uint4(0u, 0u, 0u, 0u);
            if (act) {
                const uint8_t* ap = A + kb + (lane << 4);
                if (!(((uintptr_t)ap) & 15u))
                    av = *(const uint4*)ap;
                else {
                    uint8_t* axw = (uint8_t*)&av;
#pragma unroll
                    for (int t = 0; t < 16; ++t) axw[t] = ap[t];
                }
            }
        };
        auto load_span = [&](int it, uint2& b8, float& sc) {
            const int kb  = it << 9;
            const int rem = (it < nspan) ? 512 : tail;
            const bool act = (lane << 4) < rem;
            b8 = make_uint2(0u, 0u);
            if (act) {
                const uint8_t* bp = crow + (kb >> 1) + (lane << 3);
                if (!(((uintptr_t)bp) & 7u))
                    b8 = *(const uint2*)bp;
                else {                                   /* 4-odd B (rare): two dwords */
                    b8.x = *(const uint32_t*)bp;
                    b8.y = *(const uint32_t*)(bp + 4);
                }
            }
            const int p    = lane >> 1;              /* this pair's slot inside the span */
            const bool gact = (p << 5) < rem;
            const int  g   = (it << 4) + (gact ? p : ((rem >> 5) - 1));
            sc = 0.0f;
            if (gact) {
                const uint8_t x = brow[g];
                sc = (x == 0xffu) ? __uint_as_float(0x7fc00000u)
                   : (x == 0x00u) ? 0x1p-127f
                                  : __uint_as_float((uint32_t)x << 23);
            }
        };
        auto consume_span = [&](int it, const uint2& b8, float sc) {
            const int rem = (it < nspan) ? 512 : tail;
            const bool act = (lane << 4) < rem;
            const int p    = lane >> 1;
            const bool gact = (p << 5) < rem;
            const int  g   = (it << 4) + (gact ? p : ((rem >> 5) - 1));
            uint4 av;
            load_act(it, av);
            float s = 0.0f;
            if (act) {
                const uint8_t* ax = (const uint8_t*)&av;
#pragma unroll
                for (int t = 0; t < 16; ++t) {
                    const uint32_t nib = (t < 8 ? b8.x : b8.y) >> (4 * (t & 7));
                    s = __fmaf_rn(fp8e4m3_gemv(ax[t]), L16[nib & 15u], s);
                }
            }
            s += __shfl_xor(s, 1);                   /* the pair owns group g */
            const float sa = AS[g >> 2];
            float c = gact ? (s * sa) * sc : 0.0f;
#pragma unroll
            for (int off = 2; off < kWave; off <<= 1)
                c += __shfl_xor(c, off);
            a0 += c;
        };

        struct M4Win { uint2 b8; float sc; };
        for (int gbase = 0; gbase < ntot; gbase += W) {
            const int remw = (ntot - gbase < W) ? (ntot - gbase) : W;
            M4Win win[W];
            if constexpr (W > 1) {
#pragma unroll
                for (int d = 0; d < W; ++d)
                    if (d < remw) load_span(gbase + d, win[d].b8, win[d].sc);
            }
            for (int d = 0; d < remw; ++d) {
                const int it = gbase + d;
                if constexpr (W > 1) consume_span(it, win[d].b8, win[d].sc);
                else {
                    uint2 b8; float sc;
                    load_span(it, b8, sc);
                    consume_span(it, b8, sc);
                }
            }
        }
        if (lane == 0 && r < N) Y[r] = f32_to_bf16_m4(a0);
        return;
    }

    /* MT >= 2 leg, PF (issue #7): window the per-span dword + scale byte the same
     * way as the M==1 leg. The xs staging keeps its per-span barriers (xs is
     * single-buffered); the burst just issues W spans ahead of them, so the weight
     * stream flies across every barrier the staging must pay. */
    const int ntot2 = n256 + (partial ? 1 : 0);

    auto load_sp2 = [&](int it, uint32_t& b4, float& sc) {
        const int kb   = it << 8;
        const int rem  = (it < n256) ? 256 : 128;
        /* One aligned dword per lane: nibbles kb + lane*8 + 0..7. Nibble k lives in byte
         * k>>1 (the tile's rule), so the dword is at byte (kb>>1) + lane*4 -- indexing
         * the packed row by k DOUBLES the stride, the nibble form of the byte-pitch trap. */
        const bool act = (lane << 3) < rem;
        b4 = 0u;
        if (act) b4 = *(const uint32_t*)(crow + (kb >> 1) + (lane << 2));
        const int sg = lane >> 2;                 /* this quad's group inside the span */
        const bool qact = sg < (rem >> 5);
        const int  g32 = (it << 3) + (qact ? sg : (rem >> 5) - 1);
        sc = 0.0f;
        if (qact) {
            const uint8_t x = brow[g32];
            sc = (x == 0xffu) ? __uint_as_float(0x7fc00000u)
               : (x == 0x00u) ? 0x1p-127f
                              : __uint_as_float((uint32_t)x << 23);
        }
    };
    auto consume_sp2 = [&](int it, uint32_t b4, float sc) {
        const int rem  = (it < n256) ? 256 : 128;
        const bool act = (lane << 3) < rem;
        const int sg = lane >> 2;
        const bool qact = sg < (rem >> 5);
        const int  g32 = (it << 3) + (qact ? sg : (rem >> 5) - 1);
        float sub[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            float s = 0.0f;
            if (act) {
                const float4* xp = (const float4*)&xs[m][lane << 3];
                const float4 v0 = xp[0], v1 = xp[1];
                const float x8[8] = { v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w };
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const uint32_t nib = (b4 >> (4 * t)) & 15u;   /* low nibble = even k */
                    s = __fmaf_rn(x8[t], L16[nib], s);            /* product exact in f32 */
                }
            }
            sub[m] = s;
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
#pragma unroll
            for (int off = 1; off <= 2; off <<= 1)             /* fold the quad (k chain) */
                sub[m] += __shfl_xor(sub[m], off);
            /* Dead quads join every shfl below as exact zeros -- skipping the collective
             * with `continue` would leave offsets 4/8/16 facing absent partners. Their
             * sa/sc reads are clamped in bounds and discarded. */
            const float sa = AS[(int64_t)m * as_pitch + (g32 >> 2)];
            float c = qact ? (sub[m] * sa) * sc : 0.0f;         /* quant_dot's two-multiply order */
#pragma unroll
            for (int off = 4; off < kWave; off <<= 1)          /* fold the eight group terms.
             * NOT `off; off <<= 1`: off never reaches 0 that way -- on wave32 masks >= 32
             * are out of range, the compiler either poisons them away (looks exact) or
             * emits self-selecting shuffles that double c per surviving step (~2^27). */
                c += __shfl_xor(c, off);
            acc[m] += c;
        }
    };

    struct M4Win2 { uint32_t b4; float sc; };
    for (int gbase = 0; gbase < ntot2; gbase += W) {
        const int remw = (ntot2 - gbase < W) ? (ntot2 - gbase) : W;
        M4Win2 win[W];
        if constexpr (W > 1) {
#pragma unroll
            for (int d = 0; d < W; ++d)
                if (d < remw) load_sp2(gbase + d, win[d].b4, win[d].sc);
        }
        for (int d = 0; d < remw; ++d) {
        const int it   = gbase + d;
        const int kb   = it << 8;
        const int rem  = (it < n256) ? 256 : 128;
        const int shift = (rem == 256) ? 8 : 7;
        for (int i = threadIdx.x; i < live << shift; i += kThreads) {
            const int m = i >> shift, k = i & (rem - 1);
            xs[m][k] = fp8e4m3_gemv(A[(int64_t)m * a_pitch + kb + k]);
        }
        __syncthreads();
        if constexpr (W > 1) consume_sp2(it, win[d].b4, win[d].sc);
        else { uint32_t b4; float sc; load_sp2(it, b4, sc); consume_sp2(it, b4, sc); }
        __syncthreads();                                       /* xs is rewritten next span */
        } /* d */
    }

    if (lane == 0 && r < N) {
#pragma unroll
        for (int m = 0; m < MT; ++m)
            if (m < live) Y[(int64_t)m * y_pitch + r] = f32_to_bf16_m4(acc[m]);
    }
}

/* ---- the dot2 leg (issue #6) ---------------------------------------------------------
 *
 * The #5 register leg re-decodes the SHARED fp8 activation row inside every warp's
 * inner loop: ~24 VALU per weight byte, which is exactly the 143 GB/s wall it hit
 * (4 VALU/cy/SIMD over 96 CUs). The activation row is ONE vector per token, read by
 * hundreds of weight-row warps -- every warp re-pays the decode.
 *
 * e2m1 codes and e4m3 activations are both EXACT in f16 (mantissas 2 and 3 bits,
 * exponents and max 448 all inside f16's range), and gfx11 has V_DOT2_F32_F16
 * (VOP3P, ISA p.363). So: a tiny per-dispatch prologue converts the activation row
 * to packed f16 pairs ONCE, and the weight byte's two nibbles (low = even k, the
 * tile's own rule) are exactly one dot2 operand pair -- a 256-entry LDS gather into
 * an f16-pair LUT plus one v_dot2c per weight byte. ~2 VALU + 1 LDS per byte vs ~24.
 *
 * Products of exact-f16 operands are EXACT in f32 (5 significand bits max), so the
 * only deviation from the shipped leg is the addition order inside a lane: whether
 * v_dot2c accumulates src3 + (p0 + p1) or (src3 + p0) + p1 is not spelled out in the
 * ISA, so this leg is graded on the same 3e-02 tolerance as #5 -- the attribution
 * gate settles empirically whether it is in fact bit-exact. Everything the oracle
 * order DOES own -- the per-32-of-K group ownership (lane pair), the pair butterfly,
 * the (sub * sa) * 2^(x-127) two-multiply flush, the e8m0 0xFF-NaN/0x00-subnormal
 * rules, the warp fold -- is the #5 leg verbatim.
 *
 * The prologue's f16 conversion is exact (e4m3 embeds in f16 with no rounding,
 * subnormals included; the single e4m3 NaN becomes a quiet f16 NaN and still poisons
 * every product, like the fp32 NaN poisons the FFMA chain). A16 is the plugin's own
 * buffer, so the walk needs no odd-base arms: the prologue takes the odd-A scalars.
 */
__device__ __forceinline__ uint32_t f16_pair_m4(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);        /* .x = lo -> low half */
    uint32_t u;
    memcpy(&u, &h, 4);
    return u;
}

__device__ __forceinline__ float dot2c_f16_m4(uint32_t a, uint32_t b, float acc) {
    /* The ROCm 10.1 assembler spells the accumulate form THREE operands: dst is both
     * destination and src3, and it resolves to v_dot2acc_f32_f16 in the 16-bit VOP1
     * encoding -- no CLAMP bit exists there at all. (Measured at the first link: the
     * 4-operand forms -- bare and _e64 -- are rejected, and __ockl_fdot2 lowers to a
     * VOP3P dot2 WITH clamp. The e32-suffixed 4-operand form is rejected too.) */
    asm ("v_dot2c_f32_f16 %0, %1, %2" : "+v"(acc) : "v"(a), "v"(b));
    return acc;
}

/* e2m1 as f16 bit patterns: each negative is its positive | 0x8000 -- NOT a stride.
 * (First ship had -3,-4,-6 as 0xC400/0xC600/0xC800, which are -4,-6,-8: the table
 * is where the LUT lives, so random-code gates fail loudly while uniform-nibble
 * probes with positive codes only pass silently.) */
__device__ const uint16_t kE2M1F16[16] = {
    0x0000u, 0x3800u, 0x3C00u, 0x3E00u, 0x4000u, 0x4200u, 0x4400u, 0x4600u,
    0x8000u, 0xB800u, 0xBC00u, 0xBE00u, 0xC000u, 0xC200u, 0xC400u, 0xC600u,
};

__global__ __launch_bounds__(256) void m4_act_f16_kernel(
    const uint8_t* __restrict__ A, uint32_t* __restrict__ D, int K)
{
    const int half = K >> 1;                       /* K even: the plan gate says K % 128 */
    for (int i = blockIdx.x * 256 + threadIdx.x; i < half; i += gridDim.x * 256) {
        const uint8_t* ap = A + 2 * i;             /* M == 1: ROW 0 -- a_pitch is an offset FROM row 0 */
        if (!(((uintptr_t)ap) & 1u)) {
            const uint16_t two = *(const uint16_t*)ap;
            D[i] = f16_pair_m4(fp8e4m3_m4((uint8_t)(two & 0xffu)),
                               fp8e4m3_m4((uint8_t)(two >> 8)));
        } else {                                   /* odd A base: scalar, this kernel is tiny */
            D[i] = f16_pair_m4(fp8e4m3_m4(ap[0]), fp8e4m3_m4(ap[1]));
        }
    }
}

__global__ __launch_bounds__(kThreads) void gemm_nt_mxfp4_gemv1_f16_kernel(
    const uint32_t* __restrict__ A16, const uint8_t* __restrict__ B, uint16_t* __restrict__ Y,
    const float* __restrict__ AS, const uint8_t* __restrict__ BS,
    int N, int K,
    int64_t b_pitch, int64_t y_pitch, int64_t bs_pitch)
{
    __shared__ uint32_t LUT[256];                  /* byte of two codes -> f16 pair */
    LUT[threadIdx.x] = (uint32_t)kE2M1F16[threadIdx.x & 15u]
                     | ((uint32_t)kE2M1F16[threadIdx.x >> 4] << 16);  /* low nibble = even k */

    const int w    = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int r    = blockIdx.x * (kThreads / 32) + w;
    const int rr   = r < N ? r : N - 1;

    const uint8_t* crow = B + (int64_t)rr * b_pitch;
    const uint8_t* brow = BS + (int64_t)rr * bs_pitch;

    __syncthreads();                               /* LUT visible */

    float a0 = 0.0f;
    const int nspan = K >> 9;
    const int tail  = K & 511;                     /* K % 128 == 0 gate: 128 or 384 */
    const int nit   = nspan + (tail ? 1 : 0);

    /* Double-buffered walk (#5's documented headroom): the span's GLOBAL loads are
     * issued for it+1 before the current chain runs, so the ~600-cycle load latency
     * overlaps the dot2 chain instead of stalling every wave at every span edge.
     * (Both the #5 leg and the un-prefetched dot2 leg plateau near the same ~150
     * GB/s -- the tell was that removing 20 VALU/byte bought only +5%.) */
    const uint8_t* bp0 = crow + (lane << 3);
    const uint4*  ap0 = (const uint4*)(A16 + (lane << 3));
    bool actn = (lane << 4) < ((nspan ? 512 : tail));
    uint2  b8n = make_uint2(0u, 0u);
    uint4  av0n = make_uint4(0u, 0u, 0u, 0u), av1n = make_uint4(0u, 0u, 0u, 0u);
    if (actn) {
        const uint8_t* bp = bp0;
        if (!(((uintptr_t)bp) & 7u)) b8n = *(const uint2*)bp;
        else { b8n.x = *(const uint32_t*)bp; b8n.y = *(const uint32_t*)(bp + 4); }
        av0n = ap0[0]; av1n = ap0[1];
    }
    for (int it = 0; it < nit; ++it) {
        const int kb  = it << 9;
        const int rem = (it < nspan) ? 512 : tail;
        const bool act = actn;
        uint2  b8 = b8n;
        uint4  av0 = av0n, av1 = av1n;
        const int kbn = kb + 512;
        actn = (it + 1 < nit) && ((lane << 4) < ((it + 1 < nspan) ? 512 : tail));
        if (actn) {
            const uint8_t* bp = crow + (kbn >> 1) + (lane << 3);
            if (!(((uintptr_t)bp) & 7u)) b8n = *(const uint2*)bp;
            else { b8n.x = *(const uint32_t*)bp; b8n.y = *(const uint32_t*)(bp + 4); }
            const uint4* ap = (const uint4*)(A16 + (kbn >> 1) + (lane << 3));
            av0n = ap[0]; av1n = ap[1];
        }
        const int p     = lane >> 1;
        const bool gact = (p << 5) < rem;
        const int  g    = (it << 4) + (gact ? p : ((rem >> 5) - 1));
        float sc = 0.0f;
        if (gact) {
            const uint8_t x = brow[g];
            sc = (x == 0xffu) ? __uint_as_float(0x7fc00000u)
               : (x == 0x00u) ? 0x1p-127f
                              : __uint_as_float((uint32_t)x << 23);
        }
        float s = 0.0f;
        if (act) {
            const uint8_t* bx = (const uint8_t*)&b8;
            const uint32_t av[8] = { av0.x, av0.y, av0.z, av0.w, av1.x, av1.y, av1.z, av1.w };
#pragma unroll
            for (int t = 0; t < 8; ++t)
                s = dot2c_f16_m4(LUT[bx[t]], av[t], s);
        }
        s += __shfl_xor(s, 1);                     /* the pair owns group g */
        const float sa = AS[g >> 2];
        float c = gact ? (s * sa) * sc : 0.0f;
#pragma unroll
        for (int off = 2; off < kWave; off <<= 1)
            c += __shfl_xor(c, off);
        a0 += c;
    }
    if (lane == 0 && r < N) Y[r] = f32_to_bf16_m4(a0);
}

} /* anonymous namespace */

#else  /* non-RDNA3: host-visible no-op, the launcher never dispatches it */

namespace {
template <bool DBUF>
__global__ __launch_bounds__(kThreads) void gemm_nt_mxfp4_m64_kernel(
    const uint8_t*, const uint8_t*, uint16_t*, const float*, const uint8_t*,
    int, int, int, int, int, int64_t, int64_t, int64_t, int64_t, int64_t, int, int) {}
template <int MT, int W>
__global__ __launch_bounds__(kThreads) void gemm_nt_mxfp4_gemv_kernel(
    const uint8_t*, const uint8_t*, uint16_t*, const float*, const uint8_t*,
    int, int, int, int64_t, int64_t, int64_t, int64_t, int64_t) {}
__global__ void m4_act_f16_kernel(const uint8_t*, uint32_t*, int) {}
__global__ void gemm_nt_mxfp4_gemv1_f16_kernel(
    const uint32_t*, const uint8_t*, uint16_t*, const float*, const uint8_t*,
    int, int, int64_t, int64_t, int64_t) {}
} /* anonymous namespace */

#endif

/* ---- the entries: gemm_nt_q schema, mxfp4a8 only ------------------------------------
 *
 * Positions follow oGemmNtQ (a, a_scale, b, b_scale, y, a_sum?, b_ref?). Every refusal
 * is RAD_E_UNSUPPORTED so the engine may answer from another row or the reference --
 * this is a format gate first, a shape gate second. The plan is shared with the decode
 * row below: one transcription of the oracle's format rules, two launch shapes. */

namespace {

struct M4Plan {
    const uint8_t *A, *B;
    const float*  AS;
    const uint8_t* BS;
    int64_t M, N, K;
    int as_cols, bs_cols;
    int64_t a_pitch, b_pitch, y_pitch, as_pitch, bs_pitch;
};

/* Everything both mxfp4a8 rows check before either launches. M must already be derived
 * from numel(y)/N by the caller (the arena-capacity rule, r3d_gemm_bf16.hip's comment
 * spells the 0.35 tok/s incident) and N taken from the schema parameter. */
int m4_plan(const RadArgs* a, const RadTensor* A, const RadTensor* AS,
            const RadTensor* B, const RadTensor* BS, const RadTensor* Y,
            int64_t M, int64_t N, M4Plan* P) {
    const char* dtype = rad_args_gets(a, "dtype");
    if (!dtype || strcmp(dtype, "mxfp4a8") != 0) return RAD_E_UNSUPPORTED;

    long long pb1 = 0, pb2 = 0;
    if (rad_args_geti(a, "pb1", &pb1) && pb1 > 0) return RAD_E_UNSUPPORTED;   /* no parts */
    if (rad_args_geti(a, "pb2", &pb2) && pb2 > 0) return RAD_E_UNSUPPORTED;

    if (A->dtype != RAD_F8E4M3 || B->dtype != RAD_FP4E2M1 || Y->dtype != RAD_BF16 ||
        AS->dtype != RAD_F32 || (BS->dtype != RAD_E8M0 && BS->dtype != RAD_U8))
        return RAD_E_UNSUPPORTED;

    long long K = 0;
    if (!rad_args_geti(a, "K", &K)) return RAD_E_SHAPE;
    if (N <= 0 || K <= 0) return RAD_E_SHAPE;
    if (M <= 0) return RAD_E_SHAPE;
    if (M > 0x7ffffffLL || N > 0x7ffffffLL || K > 0x7ffffffLL) return RAD_E_SHAPE;
    if (rad_tensor_numel(A) < M * K || rad_tensor_numel(B) < N * K) return RAD_E_SHAPE;

    /* Scale grids derived from the operands (make_scales' rule), then pinned to what the
     * vocabulary spells: b 32, a 128. Anything else is not this format. */
    if (BS->rank != 2 || BS->stride[1] != 1 || (long long)BS->shape[0] < N) return RAD_E_UNSUPPORTED;
    if (AS->rank != 2 || AS->stride[1] != 1 || (long long)AS->shape[0] < M) return RAD_E_UNSUPPORTED;
    const int64_t bs_cols = BS->shape[1], as_cols = AS->shape[1];
    const int bg = (int)m4_grid_block(K, bs_cols);
    const int ag = (int)m4_grid_block(K, as_cols);
    if (bg != 32 || ag != 128) return RAD_E_UNSUPPORTED;
    if (bs_cols != (K + bg - 1) / bg || as_cols != (K + ag - 1) / ag) return RAD_E_UNSUPPORTED;
    if (rad_tensor_numel(AS) < M * as_cols) return RAD_E_SHAPE;

    /* Contiguous innermost axes; the nibble rows are byte-packed exactly (a sub-byte
     * row pitch other than K/2 bytes has no defined walk here). */
    if (A->rank >= 2 && A->stride[A->rank - 1] != 1) return RAD_E_STRIDE;
    if (B->rank >= 2 && B->stride[B->rank - 1] != 1) return RAD_E_STRIDE;
    if (Y->rank >= 2 && Y->stride[Y->rank - 1] != 1) return RAD_E_STRIDE;
    if ((K & 1) || m4_row_stride(B, K) != K) return RAD_E_STRIDE;

    /* Chip gate (runtime, not compile-time: the host pass sees no __gfxNNNN__ macro). */
    static int allowed = -1;
    if (allowed < 0) {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess) dev = 0;
        hipDeviceProp_t p{};
        allowed = (hipGetDeviceProperties(&p, dev) == hipSuccess) &&
                  (strstr(p.gcnArchName, "gfx1100") || strstr(p.gcnArchName, "gfx1101") ||
                   strstr(p.gcnArchName, "gfx1102") || strstr(p.gcnArchName, "gfx1103"));
    }
    if (!allowed) return RAD_E_UNSUPPORTED;

    P->A = (const uint8_t*)A->data; P->B = (const uint8_t*)B->data;
    P->AS = (const float*)AS->data; P->BS = (const uint8_t*)BS->data;
    P->M = M; P->N = N; P->K = K;
    P->as_cols = (int)as_cols; P->bs_cols = (int)bs_cols;
    P->a_pitch  = m4_row_stride(A, K);     /* fp8: elements == bytes */
    P->b_pitch  = K / 2;                   /* validated packed above */
    P->y_pitch  = m4_row_stride(Y, N);
    P->as_pitch = m4_row_stride(AS, as_cols);
    P->bs_pitch = m4_row_stride(BS, bs_cols);
    return RAD_OK;
}

} /* anonymous namespace */

static int r3d_gemm_nt_q_impl(const RadArgs*, RadStream);
extern "C" int r3d_rad_gemm_nt_q(const RadArgs* a, RadStream s)
    R3D_DUMP_WRAP("gemm_nt_q", r3d_gemm_nt_q_impl)
static int r3d_gemm_nt_q_impl(const RadArgs* a, RadStream s) {
    static const bool no_mxfp4 = getenv("R3D_NO_MXFP4") != nullptr;
    if (no_mxfp4) return RAD_E_UNSUPPORTED;

    const RadTensor* A    = rad_arg_in(a, 0);
    const RadTensor* AS   = rad_arg_in(a, 1);
    const RadTensor* B    = rad_arg_in(a, 2);
    const RadTensor* BS   = rad_arg_in(a, 3);
    const RadTensor* Y    = rad_arg_in(a, 4);
    const RadTensor* ASUM = rad_arg_in(a, 5);
    /* b_ref (6) is accepted beside an e2m1 weight and ignored, exactly as both host
     * libraries do; this tile derives its own per-row reference at block entry. */
    if (!A || !AS || !B || !BS || !Y) return RAD_E_UNSUPPORTED;
    if (ASUM) return RAD_E_UNSUPPORTED;        /* mxfp4a8 has no zero point to remove */
    if (Y->dtype != RAD_BF16) return RAD_E_UNSUPPORTED;

    long long N = 0;
    if (!rad_args_geti(a, "N", &N) || N <= 0) return RAD_E_SHAPE;
    const long long Yn = rad_tensor_numel(Y);
    if (Yn <= 0 || Yn % N) return RAD_E_SHAPE;

    M4Plan P{};
    int rc = m4_plan(a, A, AS, B, BS, Y, Yn / N, N, &P);
    if (rc) return rc;

    static const bool no_dbuf = getenv("R3D_GEMM_NO_DBUF") != nullptr;

    hipStream_t stream = (hipStream_t)(uintptr_t)s;
    dim3 grid((unsigned)((P.N + kNB - 1) / kNB), (unsigned)((P.M + kMB - 1) / kMB));
    if (no_dbuf)
        hipLaunchKernelGGL((gemm_nt_mxfp4_m64_kernel<false>), grid, dim3(kThreads), 0, stream,
                           P.A, P.B, (uint16_t*)Y->data, P.AS, P.BS,
                           (int)P.M, (int)P.N, (int)P.K, P.as_cols, P.bs_cols,
                           P.a_pitch, P.b_pitch, P.y_pitch, P.as_pitch, P.bs_pitch, 128, 32);
    else
        hipLaunchKernelGGL((gemm_nt_mxfp4_m64_kernel<true>), grid, dim3(kThreads), 0, stream,
                           P.A, P.B, (uint16_t*)Y->data, P.AS, P.BS,
                           (int)P.M, (int)P.N, (int)P.K, P.as_cols, P.bs_cols,
                           P.a_pitch, P.b_pitch, P.y_pitch, P.as_pitch, P.bs_pitch, 128, 32);
    const hipError_t le = hipGetLastError();
    if (le != hipSuccess) fprintf(stderr, "libr3d: mxfp4 m64 launch FAILED (%s)\n",
                                  hipGetErrorName(le));
    else if (getenv("R3D_DEBUG_GEMM")) {
        static bool once = false;
        if (!once) { once = true; fprintf(stderr, "libr3d: mxfp4 WMMA tile launched (%s)\n",
                                          no_dbuf ? "single-buffer" : "dbuf pipeline"); }
    }
    return le == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* ---- gemm_nt_q at mxfp4a8, the decode band (issue #5) --------------------------------
 *
 * Same plan, same oracle order; the row's constraint band is M <= 8 so the resolver
 * hands it the decode shapes and the m64 row keeps prefill. R3D_NO_FP4_DEC=1 refuses
 * the row outright (the resolver then gives the band to the m64 tile); the refusal is
 * ALSO enforced at the registration gate in r3d_rows.cpp, which is where the engine
 * actually re-resolves. The dword-per-lane walk pins B's 4-byte alignment; anything odd
 * rides the tile. b_ref is accepted and ignored like the tile's (the fp8 decode row
 * refuses it -- at the fp8 encoding ref refuses b_ref at all, at mxfp4a8 it is a legal
 * derived plane both host libraries ignore). */
static int r3d_gemm_nt_q_m4dec_impl(const RadArgs*, RadStream);
extern "C" int r3d_rad_gemm_nt_q_mxfp4_decode(const RadArgs* a, RadStream s)
    R3D_DUMP_WRAP("gemm_nt_q_mxfp4_decode", r3d_gemm_nt_q_m4dec_impl)
static int r3d_gemm_nt_q_m4dec_impl(const RadArgs* a, RadStream s) {
    static const bool no_mxfp4 = getenv("R3D_NO_MXFP4") != nullptr ||
                                 getenv("R3D_NO_FP4_DEC") != nullptr;
    if (no_mxfp4) return RAD_E_UNSUPPORTED;

    const RadTensor* A    = rad_arg_in(a, 0);
    const RadTensor* AS   = rad_arg_in(a, 1);
    const RadTensor* B    = rad_arg_in(a, 2);
    const RadTensor* BS   = rad_arg_in(a, 3);
    const RadTensor* Y    = rad_arg_in(a, 4);
    const RadTensor* ASUM = rad_arg_in(a, 5);
    /* b_ref (6) accepted and ignored, exactly as the tile entry does. */
    if (!A || !AS || !B || !BS || !Y) return RAD_E_UNSUPPORTED;
    if (ASUM) return RAD_E_UNSUPPORTED;
    if (Y->dtype != RAD_BF16) return RAD_E_UNSUPPORTED;

    long long N = 0;
    if (!rad_args_geti(a, "N", &N) || N <= 0) return RAD_E_SHAPE;
    const long long Yn = rad_tensor_numel(Y);
    if (Yn <= 0 || Yn % N) return RAD_E_SHAPE;
    if (Yn / N > 8) return RAD_E_UNSUPPORTED;       /* the band's own bound, re-checked */

    M4Plan P{};
    int rc = m4_plan(a, A, AS, B, BS, Y, Yn / N, N, &P);
    if (rc) return rc;
    if (P.K % 128) return RAD_E_UNSUPPORTED;       /* the spans must close (the row gate) */
    if (P.b_pitch % 4 || ((uintptr_t)P.B & 3)) return RAD_E_UNSUPPORTED;

    const int mt = P.M >= 8 ? 8 : P.M >= 4 ? 4 : P.M >= 2 ? 2 : 1;
    hipStream_t stream = (hipStream_t)(uintptr_t)s;

    /* ---- dot2 leg (issue #6): M == 1 only. OPT-IN (R3D_FP4_DOT2=1): measured on the
     * ctx-512 serve A/B, one binary one env apart, it is +5% at best (pair medians
     * BASE 22.6/23.0 vs DOT2 24.2), 0% at ctx-3.3k -- the leg is correct and gated
     * but the GEMV wall is NOT instruction issue (24->2.6 VALU/byte bought 5%) and
     * not span latency (double-buffering bought 0%). Default-off keeps the shipped
     * #5 behaviour byte-identical until profiling finds the real bound. The A16 plane
     * is the plugin's OWN buffer: one lazy 1 MiB allocation, never freed; failure
     * falls back to the #5 leg -- never a mid-band refusal (the issue #4 lesson).
     * 1 MiB of VRAM is invisible to the engine's budget; the serve headroom is the
     * room it lands in. */
    static const bool dot2_on = getenv("R3D_FP4_DOT2") != nullptr;
    static uint32_t* a16 = nullptr;
    static bool a16_dead = false;
    if (mt == 1 && dot2_on && !a16_dead && P.K <= 524288) {
        if (!a16) {
            if (hipMalloc(&a16, 1u << 20) != hipSuccess) { a16 = nullptr; a16_dead = true; }
        }
        if (a16) {
            const dim3 pgrid((unsigned)((P.K / 2 + 255) / 256));
            hipLaunchKernelGGL(m4_act_f16_kernel, pgrid, dim3(256), 0, stream,
                               P.A, a16, (int)P.K);
            dim3 grid((unsigned)((P.N + 7) / 8));
            hipLaunchKernelGGL(gemm_nt_mxfp4_gemv1_f16_kernel, grid, dim3(kThreads),
                               0, stream, a16, P.B, (uint16_t*)Y->data, P.AS, P.BS,
                               (int)P.N, (int)P.K, P.b_pitch, P.y_pitch, P.bs_pitch);
            const hipError_t le = hipGetLastError();
            if (le != hipSuccess) fprintf(stderr, "libr3d: mxfp4 dot2 leg launch FAILED (%s)\n",
                                          hipGetErrorName(le));
            else if (getenv("R3D_DEBUG_GEMM")) {
                static bool once = false;
                if (!once) { once = true; fprintf(stderr, "libr3d: mxfp4 dot2 leg launched\n"); }
            }
            return le == hipSuccess ? RAD_OK : RAD_E_DEVICE;
        }
    }

    dim3 grid((unsigned)((P.N + 7) / 8));
    static const bool pf4 = getenv("R3D_FP4_GEMV_PF") != nullptr &&
                            getenv("R3D_NO_GEMV_PF") == nullptr;        /* opt-in, issue #7 */
    #define R3D_M4_GEMV(MT_, W_)                                                 \
        hipLaunchKernelGGL((gemm_nt_mxfp4_gemv_kernel<MT_, W_>), grid,           \
                           dim3(kThreads),                                        \
                           0, stream, P.A, P.B, (uint16_t*)Y->data, P.AS, P.BS,   \
                           (int)P.M, (int)P.N, (int)P.K,                          \
                           P.a_pitch, P.b_pitch, P.y_pitch, P.as_pitch, P.bs_pitch)
    #define R3D_M4_GEMV_SW(W_)                            \
        switch (mt) {                                      \
            case 1: R3D_M4_GEMV(1, W_); break;             \
            case 2: R3D_M4_GEMV(2, W_); break;             \
            case 4: R3D_M4_GEMV(4, W_); break;             \
            default: R3D_M4_GEMV(8, W_); break;            \
        }
    if (pf4) { R3D_M4_GEMV_SW(4) } else { R3D_M4_GEMV_SW(1) }
    #undef R3D_M4_GEMV_SW
    #undef R3D_M4_GEMV
    const hipError_t le = hipGetLastError();
    if (le != hipSuccess) fprintf(stderr, "libr3d: mxfp4 decode gemv launch FAILED (%s)\n",
                                  hipGetErrorName(le));
    else if (getenv("R3D_DEBUG_GEMM")) {
        static bool once = false;
        if (!once) { once = true; fprintf(stderr, "libr3d: mxfp4 decode gemv launched (mt=%d)\n", mt); }
    }
    return le == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
