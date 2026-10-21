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

} /* anonymous namespace */

#else  /* non-RDNA3: host-visible no-op, the launcher never dispatches it */

namespace {
template <bool DBUF>
__global__ __launch_bounds__(kThreads) void gemm_nt_mxfp4_m64_kernel(
    const uint8_t*, const uint8_t*, uint16_t*, const float*, const uint8_t*,
    int, int, int, int, int, int64_t, int64_t, int64_t, int64_t, int64_t, int, int) {}
} /* anonymous namespace */

#endif

/* ---- the entry: gemm_nt_q schema, mxfp4a8 only --------------------------------------
 *
 * Positions follow oGemmNtQ (a, a_scale, b, b_scale, y, a_sum?, b_ref?). Every refusal
 * is RAD_E_UNSUPPORTED so the engine may answer from another row or the reference --
 * this is a format gate first, a shape gate second. */
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

    const char* dtype = rad_args_gets(a, "dtype");
    if (!dtype || strcmp(dtype, "mxfp4a8") != 0) return RAD_E_UNSUPPORTED;

    long long pb1 = 0, pb2 = 0;
    if (rad_args_geti(a, "pb1", &pb1) && pb1 > 0) return RAD_E_UNSUPPORTED;   /* no parts */
    if (rad_args_geti(a, "pb2", &pb2) && pb2 > 0) return RAD_E_UNSUPPORTED;

    if (A->dtype != RAD_F8E4M3 || B->dtype != RAD_FP4E2M1 || Y->dtype != RAD_BF16 ||
        AS->dtype != RAD_F32 || (BS->dtype != RAD_E8M0 && BS->dtype != RAD_U8))
        return RAD_E_UNSUPPORTED;

    long long N = 0, K = 0;
    if (!rad_args_geti(a, "N", &N) || !rad_args_geti(a, "K", &K)) return RAD_E_SHAPE;
    if (N <= 0 || K <= 0) return RAD_E_SHAPE;

    /* M is numel(Y)/N -- NEVER the schema's M parameter (the arena-capacity rule,
     * r3d_gemm_bf16.hip's comment spells the 0.35 tok/s incident). */
    const long long Yn = rad_tensor_numel(Y);
    if (Yn <= 0 || Yn % N) return RAD_E_SHAPE;
    const long long M = Yn / N;
    if (M > 0x7ffffffLL || N > 0x7ffffffLL || K > 0x7ffffffLL) return RAD_E_SHAPE;
    if (rad_tensor_numel(A) < M * K || rad_tensor_numel(B) < N * K) return RAD_E_SHAPE;

    /* Scale grids derived from the operands (make_scales' rule), then pinned to what the
     * vocabulary spells: b 32, a 128. Anything else is not this tile's format. */
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

    static const bool no_dbuf = getenv("R3D_GEMM_NO_DBUF") != nullptr;

    const int64_t a_pitch  = m4_row_stride(A, K);     /* fp8: elements == bytes */
    const int64_t b_pitch  = K / 2;                   /* validated packed above */
    const int64_t y_pitch  = m4_row_stride(Y, N);
    const int64_t as_pitch = m4_row_stride(AS, as_cols);
    const int64_t bs_pitch = m4_row_stride(BS, bs_cols);

    hipStream_t stream = (hipStream_t)(uintptr_t)s;
    dim3 grid((unsigned)((N + kNB - 1) / kNB), (unsigned)((M + kMB - 1) / kMB));
    if (no_dbuf)
        hipLaunchKernelGGL((gemm_nt_mxfp4_m64_kernel<false>), grid, dim3(kThreads), 0, stream,
                           (const uint8_t*)A->data, (const uint8_t*)B->data, (uint16_t*)Y->data,
                           (const float*)AS->data, (const uint8_t*)BS->data,
                           (int)M, (int)N, (int)K, (int)as_cols, (int)bs_cols,
                           a_pitch, b_pitch, y_pitch, as_pitch, bs_pitch, ag, bg);
    else
        hipLaunchKernelGGL((gemm_nt_mxfp4_m64_kernel<true>), grid, dim3(kThreads), 0, stream,
                           (const uint8_t*)A->data, (const uint8_t*)B->data, (uint16_t*)Y->data,
                           (const float*)AS->data, (const uint8_t*)BS->data,
                           (int)M, (int)N, (int)K, (int)as_cols, (int)bs_cols,
                           a_pitch, b_pitch, y_pitch, as_pitch, bs_pitch, ag, bg);
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
