/* r3d_gemm_fp8a8.hip -- gemm_nt_q at the `fp8a8` encoding on RDNA3.
 *
 * Y[M,N] bf16 = A[M,K] fp8_e4m3 @ B[N,K]^T fp8_e4m3, block-scaled (the checkpoint
 * format behind every `quant_method: fp8, weight_block_size: [128,128]` release,
 * arch/common/rad_fp8.h):
 *   a_scale  f32  [M, ceil(K/128)]      per (row, 128 of K), written by quant_act_fp8
 *   b_scale  bf16 [ceil(N/128)][ceil(K/128)]  one scale per 128x128 weight tile
 *
 * ref_gemm.h quant_dot computes, per 128-of-K group g: sub_g = sum_{k in g} a_k*b_k
 * (ascending fp32 of exact products -- e4m3 decodes are exact in bf16), then
 *   acc += (sub_g * sa[m, g]) * sb[n/128, g]
 * and this tile reproduces that flush structure exactly: WMMA bf16 fragments hold the
 * decoded codes (exact, like the mxfp4 tile's e2m1 staging), one 128-K group of k runs
 * with untouched fp32 accumulators, and the flush applies the activation scale then the
 * weight-block scale as two separate multiplies in the oracle's order.
 *
 * Unlike the mxfp4 tile, the weight scale rides the flush, NOT the fragment: bf16 block
 * scales are ordinary floats (amax/448 quotients), not powers of two, so the bf16
 * bit-space exponent fold is not available. They are constant across the whole 64-column
 * tile (64 divides 128) and per group, so the cost is one multiply per output per group.
 * A per-row scale grid (b_scale [N, ceil(K/128)], rblk 1 -- the w8a8 spelling ref's
 * make_scales also accepts) reads the same flush with srow = n; both grids are served.
 *
 * a_sum is refused (the fp8a8 grid has no zero point; ref refuses it too), and so is
 * b_ref (ref accepts it beside an e2m1 weight only). pb1/pb2 parts are refused: ref's
 * parts path is e2m1-only.
 *
 * The flush accumulator lives in LDS for the same reason the mxfp4 tile's does (no
 * coordinate fragment in this ROCm's rocwmma); the padding columns of Ps multiply a
 * CLAMPED scale row so no flush ever reads b_scale out of bounds, and out-of-range
 * staging is plain zero, never a decode of a stray byte.
 *
 * -ffp-contract=off and the wave32 pin live in CMakeLists (plugin-wide). DBUF=1 (the
 * default) pipelines the next slice's global loads into registers while WMMA runs, A
 * double-buffered, B single (the bf16 m64's measured rule); R3D_GEMM_NO_DBUF=1 takes the
 * serialized arm; R3D_NO_FP8A8=1 refuses the row entirely so the engine answers from the
 * host reference -- the serve A/B kill switch.
 */

#include <hip/hip_runtime.h>
#include "r3d_dump.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rad_abi.h"
#include "rad_plugin.h"   /* rad_arg_in / rad_tensor_numel (host-side entry helpers) */

/* One 128-of-K scale group per WMMA burst; 256 threads band the epilogue rows. */
constexpr int kEpiThreads = 256;
constexpr int kEpiWarps   = kEpiThreads / 32;

namespace {

/* ref_gemm.h grid_block, verbatim (the mxfp4 TU's m4_ twin): the block size a scale
 * grid of `blocks` columns implies over K. */
int64_t f8_grid_block(int64_t ext, int64_t blocks) {
    if (ext <= 0) return 1;
    if (blocks <= 1) return ext;
    const int64_t lo = (ext + blocks - 1) / blocks;
    int64_t b = 1;
    while (b < lo) b <<= 1;
    return (ext + b - 1) / b == blocks ? b : lo;
}

/* ref_gemm.h row_stride: fold trailing dims until they span `cols`; the next axis out
 * is the row. */
int64_t f8_row_stride(const RadTensor* t, int64_t cols) {
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

__device__ __forceinline__ uint16_t f32_to_bf16_f8(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* rad_plugin.h rad_fp8e4m3_to_f32, ported verbatim -- the oracle's own decode (the same
 * body the mxfp4 and logits TUs carry; see those headers for the why). */
__device__ __forceinline__ float fp8e4m3_f8(uint8_t b) {
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

/* Loop-free e4m3 decode, bit-equivalent to fp8e4m3_f8 above (the subnormal branch is
 * m * 2^-9 exactly; same arithmetic form the logits GEMV ships). No per-byte divergent
 * while-loop: this sits in the bandwidth row's hot load path. */
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

/* One 64x64 output tile per workgroup, K in 16-wide slices, flushed per 128-of-K scale
 * group through LDS. Both planes stage DECODED; the two scales ride the flush. */
template <bool DBUF>
__global__ __launch_bounds__(kThreads) void gemm_nt_fp8a8_m64_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, uint16_t* __restrict__ Y,
    const float* __restrict__ AS, const uint16_t* __restrict__ BS,
    int M, int N, int K, int as_cols, int bs_cols,
    int64_t a_pitch, int64_t b_pitch,        /* e4m3: elements == bytes */
    int64_t y_pitch, int64_t as_pitch, int64_t bs_pitch,
    int rblk) {                              /* operand rows per b_scale row: 128 or 1 */

    __shared__ uint16_t As[DBUF ? 2 : 1][kMB][kKB + kLdsPad];
    __shared__ uint16_t Bs[kNB][kKB + kLdsPad];
    __shared__ float    Acc[kMB][kNB + kLdsPad];   /* persistent, per-group-flushed */
    __shared__ float    Ps[kMB][kNB + kLdsPad];    /* per-flush WMMA partial staging */
    __shared__ uint16_t Lut4m3[256];               /* e4m3 byte -> exact bf16 bits */

    const int m0 = blockIdx.y * kMB;
    const int n0 = blockIdx.x * kNB;
    const int w = threadIdx.x / kWave;
    const int tw = w >> 1;
    const int tc = (w & 1) * 2;

    if (threadIdx.x < 256)
        Lut4m3[threadIdx.x] = f32_to_bf16_f8(fp8e4m3_f8((uint8_t)threadIdx.x));
    for (int i = threadIdx.x; i < kMB * (kNB + kLdsPad); i += kThreads)
        Acc[i / (kNB + kLdsPad)][i % (kNB + kLdsPad)] = 0.0f;
    __syncthreads();                        /* LUT and Acc visible */

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bf16, rocwmma::row_major> fa;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bf16, rocwmma::col_major> fb0, fb1;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fc0, fc1;
    rocwmma::fill_fragment(fc0, 0.0f);
    rocwmma::fill_fragment(fc1, 0.0f);

    const int c_col = threadIdx.x % kKB;
    const int r_row = threadIdx.x / kKB;

    uint16_t areg[kItems], breg[kItems];

    auto prefetch = [&](int k0) {
        #pragma unroll
        for (int j = 0; j < kItems; ++j) {
            const int i = threadIdx.x + j * kThreads;
            const int ra = i / kKB, ka = k0 + i % kKB;   /* ka inside a 16-wide slice */
            const int ma = m0 + ra;
            areg[j] = (ma < M && ka < K)
                ? Lut4m3[A[(int64_t)ma * a_pitch + ka]] : (uint16_t)0;
            const int nb = n0 + ra;
            breg[j] = (nb < N && ka < K)
                ? Lut4m3[B[(int64_t)nb * b_pitch + ka]] : (uint16_t)0;
        }
    };
    auto stage_a = [&](int ab) {
        #pragma unroll
        for (int j = 0; j < kItems; ++j)
            As[ab][r_row + j * (kThreads / kKB)][c_col] = areg[j];
    };
    auto stage_b = [&]() {
        #pragma unroll
        for (int j = 0; j < kItems; ++j)
            Bs[r_row + j * (kThreads / kKB)][c_col] = breg[j];
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

    /* One 128-K group's partial: stage the WMMA accumulators to LDS, fold
     * (Ps * sa[m]) * sb[n] per oracle order, and bank into Acc. Called at uniform ks,
     * so barrier counts never diverge across the block. The scale row is clamped for
     * padding columns (their products are exact zeros and never stored, but the load
     * must still be in bounds). */
    auto flush = [&](int grp) {
        rocwmma::store_matrix_sync(&Ps[tw * 16][tc * 16],       fc0, kNB + kLdsPad, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(&Ps[tw * 16][(tc + 1) * 16], fc1, kNB + kLdsPad, rocwmma::mem_row_major);
        rocwmma::fill_fragment(fc0, 0.0f);
        rocwmma::fill_fragment(fc1, 0.0f);
        __syncthreads();                       /* Ps fully written by every wave */
        for (int i = threadIdx.x; i < kMB * kNB; i += kThreads) {
            const int r = i / kNB, c = i % kNB;
            const int m = m0 + r;
            if (m < M && grp < as_cols) {
                const int n = n0 + c;
                const int nq = n < N ? n : N - 1;
                const int srow = (rblk > 1) ? (nq / rblk) : nq;
                const float sb = __uint_as_float((uint32_t)BS[(int64_t)srow * bs_pitch + grp] << 16);
                Acc[r][c] += (Ps[r][c] * AS[(int64_t)m * as_pitch + grp]) * sb;
            }
        }
    };

    const int nks = (K + kKB - 1) / kKB;
    const int kFlush = 128 / kKB;              /* slices per scale group */

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

    for (int i = threadIdx.x; i < kMB * kNB; i += kThreads) {
        const int r = i / kNB, c = i % kNB;
        const int m = m0 + r, n = n0 + c;
        if (m < M && n < N)
            Y[(int64_t)m * y_pitch + n] = f32_to_bf16_f8(Acc[r][c]);
    }
}

/* ---- the decode-band width row (issue #4) ------------------------------------------
 *
 * The m64 tile at M = 1..8 stages every B row through a 64-row slab it uses M rows of
 * and pays the LDS round trip per slice: ~98 GB/s effective on a ~368 GB/s card. This
 * row reads every weight byte exactly once instead -- warp per weight row, four codes
 * per lane (one aligned dword, the logits GEMV's load pattern), the MT activation rows
 * staged through LDS per 128-of-K group (an fp8 row walk would be one dword per lane
 * from an L1-hot page eight times per block; staging decodes once and reads conflict-
 * free). The flush keeps the oracle's group structure: sub accumulates ascending per
 * lane over its four k, folds across the wave through a butterfly, and
 *     acc += (sub * sa[m, grp]) * sb
 * in quant_dot's order. The butterfly reassociates the k chain inside a group -- the
 * GEMM house line (r3d_gemm_bf16.hip): rel_l2, not bit-exactness, kbench adjudicates,
 * and the same fp8_9b.rkb pack that passed the tile at 3.7e-04 vs tol 3e-2 decides.
 *
 * K % 128 == 0 is the row's own constraint, so there is no group tail. Ragged N is
 * handled by clamping dead warps' load row, NOT an early return: every warp must
 * reach the block's barriers (the logits GEMV's `if (r >= N) return` is safe there
 * only because that kernel's staging loop could never miss an element for vocab rows;
 * here the clamp is the honest form).
 */
/* PF (issue #7, the MLP fix): the profile showed ~1 outstanding global load per lane
 * at MODEL shapes — the dword-per-group walk is one 128B wavefront issued immediately
 * before its FFMAs consume it. W dwords (same loads, same lane->byte mapping, so the
 * fold tree and every FP op are untouched — schedule-only, bit-exact by construction)
 * are issued back-to-back at the top of each window BEFORE the block touches LDS, so
 * all W fly across the staging barriers. Measured on the shape sweep: fp8 M==1 bands
 * gain +8..17% at W=8 (this family really was starved); W=1 is the #4 arm verbatim.
 * R3D_NO_GEMV_PF=1 selects W=1; R3D_GEMV_PF_W=16 probes a wider window (bench only). */
template <int MT, int W>
__global__ __launch_bounds__(kThreads) void gemm_nt_fp8a8_gemv_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, uint16_t* __restrict__ Y,
    const float* __restrict__ AS, const uint16_t* __restrict__ BS,
    int M, int N, int K, int as_cols, int bs_cols,
    int64_t a_pitch, int64_t b_pitch,
    int64_t y_pitch, int64_t as_pitch, int64_t bs_pitch,
    int rblk)
{
    __shared__ __align__(16) float xs[MT][128];

    const int w    = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int r    = blockIdx.x * (kThreads / 32) + w;
    const int rr   = r < N ? r : N - 1;              /* dead warps load a live row */
    const int live = M < MT ? M : MT;

    const uint8_t* crow = B + (int64_t)rr * b_pitch;
    const int srow = (rblk > 1) ? (rr / rblk) : rr;  /* same grid fold as the tile */

    for (int i = threadIdx.x; i < MT * 128; i += kThreads)
        xs[i >> 7][i & 127] = 0.0f;
    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;
    __syncthreads();

    const int ngrp = K >> 7;                          /* K % 128 == 0 is the row gate */
    const uint8_t* cwin = crow + (lane << 2);         /* this lane's dword stream base */
    for (int gbase = 0; gbase < ngrp; gbase += W) {
        const int rem = (ngrp - gbase < W) ? (ngrp - gbase) : W;
        uint32_t bwin[W];
        if constexpr (W > 1) {
            /* Window burst: W independent dwords issued back to back, none consumed
             * until its group's turn -- the MLP the profile said was missing. */
#pragma unroll
            for (int d = 0; d < W; ++d)
                bwin[d] = (d < rem) ? *(const uint32_t*)(cwin + ((gbase + d) << 7)) : 0u;
        }
        for (int d = 0; d < rem; ++d) {
        const int grp = gbase + d;
        const int kbase = grp << 7;
        for (int i = threadIdx.x; i < live * 128; i += kThreads) {
            const int m = i >> 7, k = i & 127;
            xs[m][k] = fp8e4m3_gemv(A[(int64_t)m * a_pitch + kbase + k]);
        }
        __syncthreads();

        /* One aligned dword per lane: k = lane*4 + 0..3, ascending within the lane. */
        const uint32_t b4 = (W > 1) ? bwin[d] : *(const uint32_t*)(crow + kbase + (lane << 2));
        float sub[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const float4 xv = ((const float4*)xs[m])[lane];
            const float x4[4] = { xv.x, xv.y, xv.z, xv.w };
            float s = 0.0f;
#pragma unroll
            for (int t = 0; t < 4; ++t)
                s = __fmaf_rn(x4[t], fp8e4m3_gemv((uint8_t)(b4 >> (8 * t))), s);
            sub[m] = s;
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
#pragma unroll
            for (int off = 16; off; off >>= 1)
                sub[m] += __shfl_xor(sub[m], off);
            if (m < live) {
                const float sa = AS[(int64_t)m * as_pitch + grp];
                const float sb = __uint_as_float((uint32_t)BS[(int64_t)srow * bs_pitch + grp] << 16);
                acc[m] += (sub[m] * sa) * sb;         /* quant_dot's two-multiply order */
            }
        }
        __syncthreads();                              /* xs is rewritten next group */
        } /* d */
    }

    if (lane == 0 && r < N) {
#pragma unroll
        for (int m = 0; m < MT; ++m)
            if (m < live) Y[(int64_t)m * y_pitch + r] = f32_to_bf16_f8(acc[m]);
    }
}

} /* anonymous namespace */

#else  /* non-RDNA3: host-visible no-op, the launcher never dispatches it */

namespace {
template <bool DBUF>
__global__ __launch_bounds__(kThreads) void gemm_nt_fp8a8_m64_kernel(
    const uint8_t*, const uint8_t*, uint16_t*, const float*, const uint16_t*,
    int, int, int, int, int, int64_t, int64_t, int64_t, int64_t, int64_t, int) {}
template <int MT, int W>
__global__ __launch_bounds__(kThreads) void gemm_nt_fp8a8_gemv_kernel(
    const uint8_t*, const uint8_t*, uint16_t*, const float*, const uint16_t*,
    int, int, int, int, int, int64_t, int64_t, int64_t, int64_t, int64_t, int) {}
} /* anonymous namespace */

#endif

/* ---- the gemm_nt_q_gated epilogue ---------------------------------------------------
 *
 * gated_quant_fp8_kernel ported verbatim from r3d_band.hip (the same copy pattern the
 * GDN TU's quantiser pair uses): the fused row must be byte-identical to
 * gemm_nt_q followed by gated_quant_fp8, so the epilogue is that kernel, not a
 * re-derivation. It reads the [M,N] bf16 gate_up plane the tile just computed (gate
 * half first), narrows both halves to bf16 before the gate exactly as the unfused pair
 * writes and re-reads them, and emits the fp8 code plane plus per-(row, group) scales. */

namespace {

__device__ __forceinline__ float e_sigmoid(float x) {
    return 1.0f / (1.0f + (float)exp((double)(-x)));   /* r3d_exp_f/r3d_sigmoid, verbatim */
}

__device__ __forceinline__ float e_bf16_to_f32(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}
__device__ __forceinline__ uint16_t e_f32_to_bf16(float f) {   /* same RNE as the tile's */
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* r3d_f32_to_e4m3 verbatim: RTNE, subnormal shift, saturate at 448 -- the codes the
 * oracle would have written (r3d_band.hip's rationale applies unchanged). */
__device__ __forceinline__ uint8_t e_f32_to_e4m3(float f) {
    const uint32_t u = __float_as_uint(f);
    const uint8_t  s = (uint8_t)((u >> 24) & 0x80u);
    const uint32_t a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return (uint8_t)(s | 0x7fu);     /* nan */
    if (a >= 0x43e00000u) return (uint8_t)(s | 0x7eu);    /* >= 448: clamp to the largest */
    const int e = (int)(a >> 23) - 127;
    uint32_t m = a & 0x7fffffu;
    if (e < -10) return s;                                /* below half the smallest subnormal */
    if (e < -6) {                                         /* subnormal fp8 */
        const int sh = -6 - e + 20;
        m |= 0x800000u;
        const uint32_t lo = m & ((1u << sh) - 1u);
        uint32_t q = m >> sh;
        const uint32_t half = 1u << (sh - 1);
        if (lo > half || (lo == half && (q & 1u))) ++q;
        return (uint8_t)(s | q);
    }
    uint32_t q = m >> 20;
    const uint32_t lo = m & 0xfffffu;
    int ee;
    if (lo > 0x80000u || (lo == 0x80000u && (q & 1u))) ++q;
    ee = e + 7;
    if (q == 0x8u) { q = 0; ++ee; }
    if (ee > 15 || (ee == 15 && q >= 7u)) return (uint8_t)(s | 0x7eu);
    return (uint8_t)(s | ((uint32_t)ee << 3) | q);
}

__device__ __forceinline__ float e_ldt2(const uint8_t* p, uint32_t dt, int64_t i) {
    return dt == RAD_F32 ? ((const float*)p)[i]
                         : e_bf16_to_f32(((const uint16_t*)p)[i]);
}
__device__ __forceinline__ void e_stt2(uint8_t* p, uint32_t dt, int64_t i, float v) {
    if (dt == RAD_F32) ((float*)p)[i] = v;
    else ((uint16_t*)p)[i] = e_f32_to_bf16(v);
}

/* block_max verbatim (fmaxf's NaN-skipping matches ref's strict `>` scan). */
__device__ __forceinline__ float e_block_max(float v, float* s_part) {
    for (uint32_t off = 16; off; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xFFFFFFFFull, v, off));
    const uint32_t lane = threadIdx.x & 31u, wid = threadIdx.x >> 5;
    if (lane == 0) s_part[wid] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float r = s_part[0];
        for (uint32_t w = 1; w < kEpiWarps; ++w) r = fmaxf(r, s_part[w]);
        s_part[0] = r;
    }
    __syncthreads();
    return s_part[0];
}

__device__ __forceinline__ float e_gated_h(const uint8_t* __restrict__ gub, uint32_t dt,
                                           int64_t n, int64_t i, int silu) {
    const float gt = e_ldt2(gub, RAD_BF16, i);            /* the tile's plane is always bf16 */
    const float up = e_ldt2(gub, RAD_BF16, n + i);
    const float sg = e_sigmoid(gt);
    (void)dt;
    return e_bf16_to_f32(e_f32_to_bf16((silu ? gt * sg : sg) * up));
}

__global__ __launch_bounds__(kEpiThreads) void gated_quant_epilogue_kernel(
    const uint8_t* __restrict__ gu, uint32_t dt, int64_t gu_pitch,
    uint8_t* __restrict__ q, int64_t q_pitch,
    float* __restrict__ scale, int64_t s_pitch,
    uint8_t* __restrict__ ob, uint32_t ob_dt, int64_t ob_pitch,
    int64_t n, int64_t g, int64_t ngrp, int silu) {
    __shared__ float s_part[kEpiWarps]; __shared__ float s_sc;
    const int64_t row = blockIdx.x;
    const uint8_t* __restrict__ gub = gu + row * gu_pitch;
    uint8_t* __restrict__ qr = q + row * q_pitch;
    uint8_t* __restrict__ yr = ob ? ob + row * ob_pitch : nullptr;

    for (int64_t gi = 0; gi < ngrp; ++gi) {
        const int64_t k0 = gi * g, k1 = (k0 + g < n) ? k0 + g : n;
        float amax = 0.0f;
        for (int64_t i = (int64_t)threadIdx.x; i < k1 - k0; i += (int64_t)kEpiThreads) {
            const float m = fabsf(e_gated_h(gub, dt, n, k0 + i, silu));
            if (m > amax) amax = m;
        }
        amax = e_block_max(amax, s_part);
        if (threadIdx.x == 0) {
            const float sc = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
            s_sc = sc;
            scale[row * s_pitch + gi] = sc;
        }
        __syncthreads();
        const float inv = 1.0f / s_sc;
        for (int64_t i = (int64_t)threadIdx.x; i < k1 - k0; i += (int64_t)kEpiThreads) {
            const float h = e_gated_h(gub, dt, n, k0 + i, silu);
            if (yr) e_stt2(yr, ob_dt, k0 + i, h);
            float c = h * inv;
            if (c >  448.0f) c =  448.0f;
            if (c < -448.0f) c = -448.0f;
            qr[k0 + i] = e_f32_to_e4m3(c);
        }
        __syncthreads();   /* s_sc reused next group */
    }
}

} /* anonymous namespace */

/* ---- the entries --------------------------------------------------------------------
 *
 * gemm_nt_q positions: (a, a_scale, b, b_scale, y, a_sum?, b_ref?).
 * gemm_nt_q_gated positions: (a, a_scale, b, b_scale, q, scale, out_bf16?) -- the tile
 * runs into the launch scratch as the [M,N] bf16 gate_up plane and the epilogue above
 * closes it, the byte-identical composition of the two shipped rows (libr4d's own
 * fuse-check precedent). Every refusal is RAD_E_UNSUPPORTED so the engine may answer
 * from another row or the reference -- format gate first, shape gate second. */

namespace {

struct F8Plan {
    const uint8_t *A, *B;
    const float*  AS;
    const uint16_t* BS;
    int64_t M, N, K;
    int as_cols, bs_cols, rblk;
    int64_t a_pitch, b_pitch, as_pitch, bs_pitch;
    int allowed;
};

/* Everything both fp8a8 entries share: format gate, operand dtypes, scale grids,
 * pitches, chip gate. Returns RAD_OK or the refusal code. */
int f8_plan(const RadArgs* a, const RadTensor* A, const RadTensor* AS,
            const RadTensor* B, const RadTensor* BS, int64_t M, F8Plan* P) {
    const char* dtype = rad_args_gets(a, "dtype");
    if (!dtype || strcmp(dtype, "fp8a8") != 0) return RAD_E_UNSUPPORTED;

    long long pb1 = 0, pb2 = 0;
    if (rad_args_geti(a, "pb1", &pb1) && pb1 > 0) return RAD_E_UNSUPPORTED;   /* e2m1-only parts */
    if (rad_args_geti(a, "pb2", &pb2) && pb2 > 0) return RAD_E_UNSUPPORTED;

    if (A->dtype != RAD_F8E4M3 || B->dtype != RAD_F8E4M3 ||
        AS->dtype != RAD_F32 || BS->dtype != RAD_BF16)
        return RAD_E_UNSUPPORTED;

    long long N = 0, K = 0;
    if (!rad_args_geti(a, "N", &N) || !rad_args_geti(a, "K", &K)) return RAD_E_SHAPE;
    if (N <= 0 || K <= 0) return RAD_E_SHAPE;
    if (K % 128) return RAD_E_UNSUPPORTED;   /* the scale grid must close on a whole block */
    if (M <= 0) return RAD_E_SHAPE;
    if (M > 0x7ffffffLL || N > 0x7ffffffLL || K > 0x7ffffffLL) return RAD_E_SHAPE;
    if (rad_tensor_numel(A) < M * K || rad_tensor_numel(B) < N * K) return RAD_E_SHAPE;

    /* Scale grids derived from the operands (make_scales' rules), then pinned: both are
     * 128-of-K; b_scale is either the 128-row block grid or a per-row grid. */
    if (BS->rank != 2 || BS->stride[1] != 1) return RAD_E_UNSUPPORTED;
    if (AS->rank != 2 || AS->stride[1] != 1 || (long long)AS->shape[0] < M) return RAD_E_UNSUPPORTED;
    const int64_t bs_cols = BS->shape[1], as_cols = AS->shape[1];
    const int bg = (int)f8_grid_block(K, bs_cols);
    const int ag = (int)f8_grid_block(K, as_cols);
    if (bg != 128 || ag != 128) return RAD_E_UNSUPPORTED;
    if (bs_cols != (K + 127) / 128 || as_cols != (K + 127) / 128) return RAD_E_SHAPE;
    if (rad_tensor_numel(AS) < M * as_cols) return RAD_E_SHAPE;
    int rblk;
    if ((long long)BS->shape[0] >= N) {
        if ((long long)BS->shape[0] != N) return RAD_E_SHAPE;
        rblk = 1;                                   /* per-row grid (make_scales rblk 1) */
    } else {
        const int64_t sblk = f8_grid_block(N, BS->shape[0]);
        if (sblk != 128 || (long long)BS->shape[0] != (N + 127) / 128) return RAD_E_UNSUPPORTED;
        rblk = 128;                                 /* the 128x128 checkpoint block grid */
    }
    if (rad_tensor_numel(BS) < (rblk > 1 ? (N + 127) / 128 : N) * bs_cols) return RAD_E_SHAPE;

    /* The `group` parameter must not disagree with the grid the bytes describe. */
    long long gp = 0;
    if (rad_args_geti(a, "group", &gp) && gp > 0 && gp != 128) return RAD_E_SHAPE;

    /* Contiguous innermost axes. */
    if (A->rank >= 2 && A->stride[A->rank - 1] != 1) return RAD_E_STRIDE;
    if (B->rank >= 2 && B->stride[B->rank - 1] != 1) return RAD_E_STRIDE;

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
    P->AS = (const float*)AS->data; P->BS = (const uint16_t*)BS->data;
    P->M = M; P->N = N; P->K = K;
    P->as_cols = (int)as_cols; P->bs_cols = (int)bs_cols; P->rblk = rblk;
    P->a_pitch  = f8_row_stride(A, K);
    P->b_pitch  = f8_row_stride(B, K);
    P->as_pitch = f8_row_stride(AS, as_cols);
    P->bs_pitch = f8_row_stride(BS, bs_cols);
    P->allowed = allowed;
    return RAD_OK;
}

void f8_launch_tile(const F8Plan& P, void* ydata, int64_t y_pitch, RadStream s,
                    const char* trace) {
    static const bool no_dbuf = getenv("R3D_GEMM_NO_DBUF") != nullptr;
    static const bool no_dec  = getenv("R3D_NO_FP8_DEC") != nullptr;
    static const bool no_pf   = getenv("R3D_NO_GEMV_PF") != nullptr;   /* issue #7 A/B */
    static const int  pf_w    = getenv("R3D_GEMV_PF_W") ? atoi(getenv("R3D_GEMV_PF_W")) : 8;
    hipStream_t stream = (hipStream_t)(uintptr_t)s;

    /* Decode band (issue #4): at M <= 8 the width row reads every weight byte once.
     * The dword-per-lane load pins B's alignment; anything odd rides the tile.
     * R3D_NO_FP8_DEC=1 puts both this dispatch and the plain decode row back to the
     * m64 arm -- the serve A/B kill switch for the band. */
    if (!no_dec && P.M <= 8 && (P.b_pitch % 4) == 0 && (((uintptr_t)P.B & 3) == 0)) {
        const int mt = P.M >= 8 ? 8 : P.M >= 4 ? 4 : P.M >= 2 ? 2 : 1;
        dim3 grid((unsigned)((P.N + 7) / 8));
        #define R3D_F8_GEMV(MT_, W_)                                                   \
            hipLaunchKernelGGL((gemm_nt_fp8a8_gemv_kernel<MT_, W_>), grid,             \
                               dim3(kThreads),                                         \
                               0, stream, P.A, P.B, (uint16_t*)ydata, P.AS, P.BS,      \
                               (int)P.M, (int)P.N, (int)P.K, P.as_cols, P.bs_cols,     \
                               P.a_pitch, P.b_pitch, y_pitch, P.as_pitch, P.bs_pitch,  \
                               P.rblk)
        #define R3D_F8_GEMV_SW(W_)                            \
            switch (mt) {                                     \
                case 1: R3D_F8_GEMV(1, W_); break;            \
                case 2: R3D_F8_GEMV(2, W_); break;            \
                case 4: R3D_F8_GEMV(4, W_); break;            \
                default: R3D_F8_GEMV(8, W_); break;           \
            }
        if (no_pf)        { R3D_F8_GEMV_SW(1)  }
        else if (pf_w == 16) { R3D_F8_GEMV_SW(16) }
        else              { R3D_F8_GEMV_SW(8)  }
        #undef R3D_F8_GEMV_SW
        #undef R3D_F8_GEMV
        if (getenv("R3D_DEBUG_GEMM")) {
            static bool once = false;
            if (!once) { once = true; fprintf(stderr, "libr3d: fp8a8 decode gemv launched (mt=%d)\n", mt); }
        }
        return;
    }

    dim3 grid((unsigned)((P.N + kNB - 1) / kNB), (unsigned)((P.M + kMB - 1) / kMB));
    if (no_dbuf)
        hipLaunchKernelGGL((gemm_nt_fp8a8_m64_kernel<false>), grid, dim3(kThreads), 0, stream,
                           P.A, P.B, (uint16_t*)ydata, P.AS, P.BS,
                           (int)P.M, (int)P.N, (int)P.K, P.as_cols, P.bs_cols,
                           P.a_pitch, P.b_pitch, y_pitch, P.as_pitch, P.bs_pitch, P.rblk);
    else
        hipLaunchKernelGGL((gemm_nt_fp8a8_m64_kernel<true>), grid, dim3(kThreads), 0, stream,
                           P.A, P.B, (uint16_t*)ydata, P.AS, P.BS,
                           (int)P.M, (int)P.N, (int)P.K, P.as_cols, P.bs_cols,
                           P.a_pitch, P.b_pitch, y_pitch, P.as_pitch, P.bs_pitch, P.rblk);
    const hipError_t le = hipGetLastError();
    if (le != hipSuccess) fprintf(stderr, "libr3d: %s launch FAILED (%s)\n",
                                  trace, hipGetErrorName(le));
    else if (getenv("R3D_DEBUG_GEMM")) {
        static bool once = false;
        if (!once) { once = true; fprintf(stderr, "libr3d: fp8a8 WMMA tile launched (%s)\n",
                                          no_dbuf ? "single-buffer" : "dbuf pipeline"); }
    }
}

} /* anonymous namespace */

static int r3d_gemm_nt_q_fp8_impl(const RadArgs*, RadStream);
extern "C" int r3d_rad_gemm_nt_q_fp8(const RadArgs* a, RadStream s)
    R3D_DUMP_WRAP("gemm_nt_q_fp8", r3d_gemm_nt_q_fp8_impl)
static int r3d_gemm_nt_q_fp8_impl(const RadArgs* a, RadStream s) {
    static const bool no_fp8 = getenv("R3D_NO_FP8A8") != nullptr;
    if (no_fp8) return RAD_E_UNSUPPORTED;

    const RadTensor* A    = rad_arg_in(a, 0);
    const RadTensor* AS   = rad_arg_in(a, 1);
    const RadTensor* B    = rad_arg_in(a, 2);
    const RadTensor* BS   = rad_arg_in(a, 3);
    const RadTensor* Y    = rad_arg_in(a, 4);
    const RadTensor* ASUM = rad_arg_in(a, 5);
    const RadTensor* BREF = rad_arg_in(a, 6);
    if (!A || !AS || !B || !BS || !Y) return RAD_E_UNSUPPORTED;
    if (ASUM) return RAD_E_UNSUPPORTED;   /* no zero point in this grid (ref refuses it too) */
    if (BREF) return RAD_E_UNSUPPORTED;   /* ref accepts b_ref beside an e2m1 weight only */
    if (Y->dtype != RAD_BF16) return RAD_E_UNSUPPORTED;

    /* M is numel(Y)/N -- NEVER the schema's M parameter (the arena-capacity rule). */
    long long Np = 0;
    if (!rad_args_geti(a, "N", &Np) || Np <= 0) return RAD_E_SHAPE;
    const long long Yn = rad_tensor_numel(Y);
    if (Yn <= 0 || Yn % Np) return RAD_E_SHAPE;

    if (Y->rank >= 2 && Y->stride[Y->rank - 1] != 1) return RAD_E_STRIDE;

    F8Plan P{};
    int rc = f8_plan(a, A, AS, B, BS, Yn / Np, &P);
    if (rc) return rc;
    f8_launch_tile(P, Y->data, f8_row_stride(Y, Np), s, "fp8a8 m64");
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* ---- gemm_nt_q at fp8a8, the decode band (issue #4) ---------------------------------
 *
 * Same plan, same oracle order; the row's constraint band is M <= 8 so the resolver
 * hands it the decode shapes and the m64 row keeps prefill. R3D_NO_FP8_DEC=1 refuses
 * the row outright (the resolver then gives the band to gemm_nt_q_fp8a8); the refusal
 * of a misaligned B keeps the dword-per-lane load honest -- the tile takes bytes.
 * The gated row shares the dispatch: its scratch tile rides the same GEMV at M <= 8,
 * which is what carries the FFN gate_up plane's two-thirds of decode weight traffic. */

static int r3d_gemm_nt_q_fp8_dec_impl(const RadArgs*, RadStream);
extern "C" int r3d_rad_gemm_nt_q_fp8_decode(const RadArgs* a, RadStream s)
    R3D_DUMP_WRAP("gemm_nt_q_fp8_decode", r3d_gemm_nt_q_fp8_dec_impl)
static int r3d_gemm_nt_q_fp8_dec_impl(const RadArgs* a, RadStream s) {
    static const bool no_fp8 = getenv("R3D_NO_FP8A8") != nullptr ||
                               getenv("R3D_NO_FP8_DEC") != nullptr;
    if (no_fp8) return RAD_E_UNSUPPORTED;

    const RadTensor* A    = rad_arg_in(a, 0);
    const RadTensor* AS   = rad_arg_in(a, 1);
    const RadTensor* B    = rad_arg_in(a, 2);
    const RadTensor* BS   = rad_arg_in(a, 3);
    const RadTensor* Y    = rad_arg_in(a, 4);
    const RadTensor* ASUM = rad_arg_in(a, 5);
    const RadTensor* BREF = rad_arg_in(a, 6);
    if (!A || !AS || !B || !BS || !Y) return RAD_E_UNSUPPORTED;
    if (ASUM || BREF) return RAD_E_UNSUPPORTED;      /* the plain row's refusals, same */
    if (Y->dtype != RAD_BF16) return RAD_E_UNSUPPORTED;

    long long Np = 0;
    if (!rad_args_geti(a, "N", &Np) || Np <= 0) return RAD_E_SHAPE;
    const long long Yn = rad_tensor_numel(Y);
    if (Yn <= 0 || Yn % Np) return RAD_E_SHAPE;
    if (Yn / Np > 8) return RAD_E_UNSUPPORTED;       /* the band's own bound, re-checked */

    if (Y->rank >= 2 && Y->stride[Y->rank - 1] != 1) return RAD_E_STRIDE;

    F8Plan P{};
    int rc = f8_plan(a, A, AS, B, BS, Yn / Np, &P);
    if (rc) return rc;
    if ((P.b_pitch % 4) != 0 || ((uintptr_t)P.B & 3) != 0) return RAD_E_UNSUPPORTED;
    f8_launch_tile(P, Y->data, f8_row_stride(Y, Np), s, "fp8a8 decode gemv");
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* ---- gemm_nt_q_gated at fp8a8 --------------------------------------------------------
 *
 * The tile runs into the launch scratch as the [M,N] bf16 gate_up plane (scratch sized
 * by the row's hook below), and the ported epilogue closes it into the fp8 code pair --
 * the byte-identical composition of gemm_nt_q followed by gated_quant_fp8. M is
 * numel(q)/n (the code plane is the output carrier, ref_gemm_nt_q_gated's own rule). */

namespace {

/* row2_pitch from r3d_band.hip (the sealed form: rank 0 refuses, rank >= 3 walks every
 * adjacent pair -- an innermost stride hole is the 27B gate-corruption trap). */
int f8_row2_pitch(const RadTensor* t, int64_t cols, int64_t* pitch) {
    if (!t || !t->data) return -1;
    if (t->rank == 0) return -4;
    if (t->rank == 1) {
        if (rad_tensor_numel(t) != cols || t->stride[0] != 1) return -2;
        *pitch = 0;
        return 0;
    }
    if (t->rank == 2) {
        if (t->stride[1] != 1) return -3;
        *pitch = (t->shape[1] == cols) ? t->stride[0] : cols;
        return 0;
    }
    if (t->stride[t->rank - 1] != 1) return -3;
    if (t->shape[t->rank - 1] != cols) { *pitch = cols; return 0; }   /* ref's linear fallback */
    for (int i = 0; i < (int)t->rank - 1; ++i)
        if (t->stride[i] != t->shape[i + 1] * t->stride[i + 1]) return -4;
    *pitch = cols;
    return 0;
}

} /* anonymous namespace */

static int r3d_gemm_nt_q_gated_fp8_impl(const RadArgs*, RadStream);
extern "C" int r3d_rad_gemm_nt_q_gated_fp8(const RadArgs* a, RadStream s)
    R3D_DUMP_WRAP("gemm_nt_q_gated_fp8", r3d_gemm_nt_q_gated_fp8_impl)
static int r3d_gemm_nt_q_gated_fp8_impl(const RadArgs* a, RadStream s) {
    static const bool no_fp8 = getenv("R3D_NO_FP8A8") != nullptr;
    if (no_fp8) return RAD_E_UNSUPPORTED;

    const RadTensor* A    = rad_arg_in(a, 0);
    const RadTensor* AS   = rad_arg_in(a, 1);
    const RadTensor* B    = rad_arg_in(a, 2);
    const RadTensor* BS   = rad_arg_in(a, 3);
    const RadTensor* Q    = rad_arg_in(a, 4);
    const RadTensor* SC   = rad_arg_in(a, 5);
    const RadTensor* OB   = rad_arg_in(a, 6);
    const RadTensor* ASUM = rad_arg_in(a, 7);
    if (!A || !AS || !B || !BS || !Q || !SC) return RAD_E_UNSUPPORTED;
    if (ASUM) return RAD_E_UNSUPPORTED;
    if (Q->dtype != RAD_F8E4M3 || SC->dtype != RAD_F32) return RAD_E_UNSUPPORTED;
    if (OB && OB->dtype != RAD_BF16 && OB->dtype != RAD_F32) return RAD_E_UNSUPPORTED;

    /* `act` is required here (ref_gemm_nt_q_gated's rule: an implementation may compile
     * one gate, so an optional parameter would let it resolve for the other). */
    const char* act = rad_args_gets(a, "act");
    if (!act) return RAD_E_UNSUPPORTED;
    const int silu = strcmp(act, "sigmoid") != 0;
    if (silu && strcmp(act, "silu") != 0) return RAD_E_UNSUPPORTED;

    long long N = 0;
    if (!rad_args_geti(a, "N", &N) || N <= 0 || (N & 1)) return RAD_E_SHAPE;
    const long long n = N / 2;

    long long g = 0;
    if (!rad_args_geti(a, "group", &g) || g <= 0) g = n;   /* ref's default */
    if (g > n) g = n;
    const long long ngrp = (n + g - 1) / g;

    /* M is numel(q)/n -- the code plane is the output carrier. */
    const long long Qn = rad_tensor_numel(Q);
    if (Qn <= 0 || Qn % n) return RAD_E_SHAPE;
    const long long M = Qn / n;
    if (rad_tensor_numel(SC) < M * ngrp) return RAD_E_SHAPE;
    if (OB && rad_tensor_numel(OB) < M * n) return RAD_E_SHAPE;

    F8Plan P{};
    int rc = f8_plan(a, A, AS, B, BS, M, &P);
    if (rc) return rc;
    if (P.N != N) return RAD_E_SHAPE;   /* N param and the b_scale grid must agree */

    /* The gate_up plane lands in the launch scratch: M*N bf16. The row's scratch hook
     * sizes the arena at declare for the band's M upper bound (M <= 64); refuse -- never
     * overrun -- if a launch outgrows what the arena got. */
    const int64_t need = M * N * (int64_t)sizeof(uint16_t);
    if (!a->scratch || a->scratch_bytes < need) return RAD_E_UNSUPPORTED;

    int64_t qp = 0, sp = 0, op = 0;
    if (f8_row2_pitch(Q, n, &qp)) return RAD_E_STRIDE;
    if (f8_row2_pitch(SC, ngrp, &sp)) return RAD_E_STRIDE;
    if (OB && f8_row2_pitch(OB, n, &op)) return RAD_E_STRIDE;

    hipStream_t stream = (hipStream_t)(uintptr_t)s;
    f8_launch_tile(P, a->scratch, N, s, "fp8a8 gated tile");
    if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;

    /* Pitches follow the band entry's convention: the kernel offsets the byte pointer
     * first and indexes elements from there, so the bf16 plane and the optional bf16/f32
     * product take BYTE pitches (rad_dtype_bytes), while the f32 scale takes element
     * units (it is indexed as float* directly). q is e4m3: bytes == elements. */
    gated_quant_epilogue_kernel<<<dim3((unsigned)M), dim3(kEpiThreads), 0, stream>>>(
        (const uint8_t*)a->scratch, RAD_BF16, N * (int64_t)sizeof(uint16_t),
        (uint8_t*)Q->data, qp, (float*)SC->data, sp,
        OB ? (uint8_t*)OB->data : nullptr, OB ? OB->dtype : RAD_F32,
        OB ? rad_dtype_bytes(OB->dtype, op) : 0,
        n, g, ngrp, silu);
    if (getenv("R3D_DEBUG_GEMM")) {
        static bool once = false;
        if (!once) { once = true; fprintf(stderr, "libr3d: fp8a8 gated row launched\n"); }
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* Scratch for the fused row: the [M,N] bf16 gate_up plane at the band's M upper bound.
 * The row claims M <= 64, so 64*N*2 covers every launch the resolver can offer; a wider
 * arena than the band's bound costs a few KiB and never a refusal. */
extern "C" int64_t r3d_rad_gemm_nt_q_gated_fp8_scratch(const RadArgs* a) {
    long long N = 0;
    if (!rad_args_geti(a, "N", &N) || N <= 0 || N > (1LL << 34)) return 0;   /* 0 = none */
    return 64 * N * (int64_t)sizeof(uint16_t);
}
