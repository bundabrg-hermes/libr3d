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

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rad_abi.h"
#include "rad_plugin.h"   /* rad_arg_in / rad_tensor_numel (host-side entry helpers) */

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

} /* anonymous namespace */

#else  /* non-RDNA3: host-visible no-op, the launcher never dispatches it */

namespace {
template <bool DBUF>
__global__ __launch_bounds__(kThreads) void gemm_nt_fp8a8_m64_kernel(
    const uint8_t*, const uint8_t*, uint16_t*, const float*, const uint16_t*,
    int, int, int, int, int, int64_t, int64_t, int64_t, int64_t, int64_t, int) {}
} /* anonymous namespace */

#endif

/* ---- the entry: gemm_nt_q schema, fp8a8 only ----------------------------------------
 *
 * Positions follow oGemmNtQ (a, a_scale, b, b_scale, y, a_sum?, b_ref?). Every refusal
 * is RAD_E_UNSUPPORTED so the engine may answer from another row or the reference --
 * format gate first, shape gate second. */
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

    const char* dtype = rad_args_gets(a, "dtype");
    if (!dtype || strcmp(dtype, "fp8a8") != 0) return RAD_E_UNSUPPORTED;

    long long pb1 = 0, pb2 = 0;
    if (rad_args_geti(a, "pb1", &pb1) && pb1 > 0) return RAD_E_UNSUPPORTED;   /* e2m1-only parts */
    if (rad_args_geti(a, "pb2", &pb2) && pb2 > 0) return RAD_E_UNSUPPORTED;

    if (A->dtype != RAD_F8E4M3 || B->dtype != RAD_F8E4M3 || Y->dtype != RAD_BF16 ||
        AS->dtype != RAD_F32 || BS->dtype != RAD_BF16)
        return RAD_E_UNSUPPORTED;

    long long N = 0, K = 0;
    if (!rad_args_geti(a, "N", &N) || !rad_args_geti(a, "K", &K)) return RAD_E_SHAPE;
    if (N <= 0 || K <= 0) return RAD_E_SHAPE;
    if (K % 128) return RAD_E_UNSUPPORTED;   /* the scale grid must close on a whole block */

    /* M is numel(Y)/N -- NEVER the schema's M parameter (the arena-capacity rule). */
    const long long Yn = rad_tensor_numel(Y);
    if (Yn <= 0 || Yn % N) return RAD_E_SHAPE;
    const long long M = Yn / N;
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
    if (Y->rank >= 2 && Y->stride[Y->rank - 1] != 1) return RAD_E_STRIDE;

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

    const int64_t a_pitch  = f8_row_stride(A, K);
    const int64_t b_pitch  = f8_row_stride(B, K);
    const int64_t y_pitch  = f8_row_stride(Y, N);
    const int64_t as_pitch = f8_row_stride(AS, as_cols);
    const int64_t bs_pitch = f8_row_stride(BS, bs_cols);

    hipStream_t stream = (hipStream_t)(uintptr_t)s;
    dim3 grid((unsigned)((N + kNB - 1) / kNB), (unsigned)((M + kMB - 1) / kMB));
    if (no_dbuf)
        hipLaunchKernelGGL((gemm_nt_fp8a8_m64_kernel<false>), grid, dim3(kThreads), 0, stream,
                           (const uint8_t*)A->data, (const uint8_t*)B->data, (uint16_t*)Y->data,
                           (const float*)AS->data, (const uint16_t*)BS->data,
                           (int)M, (int)N, (int)K, (int)as_cols, (int)bs_cols,
                           a_pitch, b_pitch, y_pitch, as_pitch, bs_pitch, rblk);
    else
        hipLaunchKernelGGL((gemm_nt_fp8a8_m64_kernel<true>), grid, dim3(kThreads), 0, stream,
                           (const uint8_t*)A->data, (const uint8_t*)B->data, (uint16_t*)Y->data,
                           (const float*)AS->data, (const uint16_t*)BS->data,
                           (int)M, (int)N, (int)K, (int)as_cols, (int)bs_cols,
                           a_pitch, b_pitch, y_pitch, as_pitch, bs_pitch, rblk);
    const hipError_t le = hipGetLastError();
    if (le != hipSuccess) fprintf(stderr, "libr3d: fp8a8 m64 launch FAILED (%s)\n",
                                  hipGetErrorName(le));
    else if (getenv("R3D_DEBUG_GEMM")) {
        static bool once = false;
        if (!once) { once = true; fprintf(stderr, "libr3d: fp8a8 WMMA tile launched (%s)\n",
                                          no_dbuf ? "single-buffer" : "dbuf pipeline"); }
    }
    return le == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
