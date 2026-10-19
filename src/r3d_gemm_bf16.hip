/* r3d_gemm_bf16.hip -- libr3d's `gemm_nt` on RDNA3: Y[M,N] = A[M,K] @ B[N,K]^T, bf16.
 *
 * The arithmetic contract is libref's (ref_gemm.cpp gemm_plain/gemm_tiles, Apache-2.0,
 * (c) radiance authors -- ported with attribution):
 *   * each output accumulates in f32 over ascending k from zero, from bf16-widened operands
 *     widened exactly (bf16 -> f32 is lossless);
 *   * -ffp-contract=off (CMakeLists): the product and the add stay separate roundings, as
 *     the oracle's `acc += a * b` compiles;
 *   * the single narrowing to y's dtype happens at the store (RTNE).
 * The k loop is tiled here, so the f32 partials recombine in chunks of BK rather than one
 * flat chain. Every reassociation sits below bf16 store rounding, and the oracle contract is
 * rel_l2, not bit-exactness -- kbench adjudicates, and any drift shows there.
 *
 * The tile here is the correctness-first fallback: 64x64 outputs per workgroup, K in
 * 16-slice LDS tiles, 256 threads at 4x4 outputs each, scalar bf16 loads. Dispatch:
 * decode (M=1) goes to the bandwidth-shaped GEMV row below; every M > 1 goes to the
 * WMMA prefill tile in r3d_gemm_bf16_m64.hip, and this scalar tile remains the fallback
 * for a refused/failed m64 launch (and the bit-exact twin of the oracle's k-chain for
 * desk work).
 *
 * Operand reading mirrors ref_gemm.h exactly:
 *   * row pitch is the trailing-dimension FOLD, not stride[rank-2]: a rank-3 qkv output read
 *     as [M, N] carries the fold, and using the head stride leaves later rows as exact zeros
 *     (ref's comment, and libr3d's own rope postmortem about axes vanishing from addresses);
 *   * columns must be contiguous (the engine's activations and weights are); non-unit column
 *     stride is refused at the entry point, never approximated;
 *   * B is a weight: contiguous [N, K], row pitch K (read off stride[rank-2] anyway).
 */

#include <hip/hip_runtime.h>

#include "rad_abi.h"
#include "rad_plugin.h"
#include "r3d_gemm.h"

namespace {

constexpr int kBM = 64, kBN = 64, kBK = 16;      /* output tile, k slice */
constexpr int kPad = 4;                          /* LDS row padding */
constexpr int kThreads = 256;                    /* 16x16 threads, 4x4 outputs each */

__device__ __forceinline__ float r3d_bf16_to_f32(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}
__device__ __forceinline__ uint16_t r3d_f32_to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* One workgroup's tile. A row pitch, B row pitch and Y row pitch arrive pre-folded by the
 * host (ref's row_stride rule); all three operands are k/n-contiguous by entry-point check. */
__global__ __launch_bounds__(kThreads) void gemm_nt_bf16_kernel(
    const uint16_t* __restrict__ A, const uint16_t* __restrict__ B,
    uint16_t* __restrict__ Y, float* __restrict__ Yf,
    int M, int N, int K, int64_t a_pitch, int64_t b_pitch, int64_t y_pitch) {

    __shared__ float As[kBK][kBM + kPad];
    __shared__ float Bs[kBK][kBN + kPad];

    const int m0 = blockIdx.y * kBM;
    const int n0 = blockIdx.x * kBN;
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;

    float acc[4][4] = {};

    for (int k0 = 0; k0 < K; k0 += kBK) {
        /* Load A tile (kBM x kBK) and transpose into As[k][m]; B tile (kBN x kBK) into
         * Bs[k][n]. Out-of-tile rows (ragged M/N) and columns (ragged K) load as zero, which
         * contributes nothing and is never stored. */
        for (int i = threadIdx.x; i < kBM * kBK; i += kThreads) {
            const int r = i / kBK, c = i % kBK;
            const int m = m0 + r, k = k0 + c;
            As[c][r] = (m < M && k < K) ? r3d_bf16_to_f32(A[(int64_t)m * a_pitch + k]) : 0.f;
        }
        for (int i = threadIdx.x; i < kBN * kBK; i += kThreads) {
            const int r = i / kBK, c = i % kBK;
            const int n = n0 + r, k = k0 + c;
            Bs[c][r] = (n < N && k < K) ? r3d_bf16_to_f32(B[(int64_t)n * b_pitch + k]) : 0.f;
        }
        __syncthreads();

        #pragma unroll
        for (int c = 0; c < kBK; ++c) {
            float af[4], bf[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) af[j] = As[c][ty * 4 + j];
            #pragma unroll
            for (int j = 0; j < 4; ++j) bf[j] = Bs[c][tx * 4 + j];
            #pragma unroll
            for (int r = 0; r < 4; ++r)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    acc[r][j] += af[r] * bf[j];   /* separate mul and add by flag */
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int m = m0 + ty * 4 + r;
        if (m >= M) continue;
        const int64_t yb = (int64_t)m * y_pitch;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + tx * 4 + j;
            if (n >= N) continue;
            if (Yf) Yf[yb + n] = acc[r][j];
            else    Y[yb + n]  = r3d_f32_to_bf16(acc[r][j]);
        }
    }
}

/* Decode GEMV (M == 1): Y[n] = dot(A, B[n]). The tile kernel reads every weight element
 * once per 64-row block, which at M=1 means 63/64 of each A tile's workgroup budget is
 * spent on zero rows; at a measured ~1.45 GB/s it caps decode at 0.4 tok/s on the XTX.
 * This row is bandwidth-shaped instead: one 32-lane wavefront per output row (the project
 * pins the wave32 ABI per TU -- see CMakeLists -mno-wavefrontsize64), each lane striding K
 * in 16-byte (8 x bf16) vectors, so a wavefront's loads are a contiguous 512-byte run and
 * every weight element is touched exactly once.
 *
 * Arithmetic contract, same grounds as the tile above: each lane accumulates f32 over its
 * own ascending k subsequence; the wavefront then reassociates the 32 partials through a
 * shfl_xor butterfly tree. That reassociation sits entirely above f32 partials and below
 * the single bf16 store rounding -- exactly the class of reassociation the tiling note at
 * the top of this file already sanctions, and kbench rel_l2 adjudicates. -ffp-contract=off
 * keeps mul and add separate roundings as in libref.
 */
constexpr int kGemvThreads = 256;   /* 8 wavefronts = 8 rows per workgroup */
constexpr int kGemvRows    = 8;
constexpr int kWave        = 32;    /* wave32 ABI: pinned by -mno-wavefrontsize64 in CMake */

__device__ __forceinline__ float r3d_dot8_bf16(const uint4& av, const uint4& bv, float acc) {
    const uint32_t* au = reinterpret_cast<const uint32_t*>(&av);
    const uint32_t* bu = reinterpret_cast<const uint32_t*>(&bv);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float a0 = r3d_bf16_to_f32((uint16_t)(au[i] & 0xffffu));
        const float a1 = r3d_bf16_to_f32((uint16_t)(au[i] >> 16));
        const float b0 = r3d_bf16_to_f32((uint16_t)(bu[i] & 0xffffu));
        const float b1 = r3d_bf16_to_f32((uint16_t)(bu[i] >> 16));
        acc += a0 * b0;
        acc += a1 * b1;
    }
    return acc;
}

__global__ __launch_bounds__(kGemvThreads) void gemv_nt_bf16_kernel(
    const uint16_t* __restrict__ A, const uint16_t* __restrict__ B,
    uint16_t* __restrict__ Y, float* __restrict__ Yf,
    int N, int K, int64_t b_pitch) {

    const int lane = threadIdx.x & (kWave - 1);
    const int row  = blockIdx.x * kGemvRows + (threadIdx.x / kWave);

    float acc = 0.f;
    if (row < N) {
        const uint4* __restrict__ av = reinterpret_cast<const uint4*>(A);
        const uint4* __restrict__ bp = reinterpret_cast<const uint4*>(B + (int64_t)row * b_pitch);
        const int kvec = K / 8;                       /* entry point guarantees K % 8 == 0 */
        for (int v = lane; v < kvec; v += kWave)
            acc = r3d_dot8_bf16(av[v], bp[v], acc);   /* A is one row: same vector index */
    }

    #pragma unroll
    for (int o = kWave / 2; o > 0; o >>= 1)
        acc += __shfl_xor_sync(0xffffffffffffffffull, acc, o);  /* ROCm 10.1: mask must be 64-bit */

    /* Store flat at [row], NOT row*y_pitch: y_pitch is the pitch BETWEEN M rows (the
     * ref fold), and this kernel is the M == 1 band -- the tile kernel's m = 0 store is
     * Y[0 + n] for every column n, so the matching store here is Y[row]. Scaling by
     * y_pitch walks off the end: kbench's first M=1 gemm_nt case caught exactly that
     * as a write fault ~1.4 MiB past the end of the operand pool. */
    if (row < N && lane == 0) {
        if (Yf) Yf[row] = acc;
        else    Y[row]  = r3d_f32_to_bf16(acc);
    }
}

/* ref_gemm.h row_stride: fold trailing dims until they span `cols`; the next axis out is the
 * row. Ported so a rank-3 operand reads exactly as the oracle reads it. */
int64_t row_stride(const RadTensor* t, int64_t cols) {
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

/* Launch entry; operands positional in the gemm_nt schema: a, b, y. */
/* Shared launch for gemm_nt and logits_gemm: ref_logits_gemm IS gemm_nt_impl with the
 * dimensions spelled n_vocab/n_embd (libref says so in one line), the operands are the
 * same three in the same order, and the arithmetic is the same tile. */
static int gemm_nt_launch(const RadArgs* a, RadStream s, const char* n_name, const char* k_name) {
    const RadTensor* A = rad_arg_in(a, 0);
    const RadTensor* B = rad_arg_in(a, 1);
    const RadTensor* Y = rad_arg_in(a, 2);
    if (!A || !B || !Y) return RAD_E_INVAL;
    if (A->dtype != RAD_BF16 || B->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (Y->dtype != RAD_BF16 && Y->dtype != RAD_F32) return RAD_E_DTYPE;

    long long M = 0, N = 0, K = 0;
    if (!rad_args_geti(a, "M", &M) || !rad_args_geti(a, n_name, &N) || !rad_args_geti(a, k_name, &K))
        return RAD_E_SHAPE;
    if (N <= 0 || K <= 0) return RAD_E_SHAPE;

    /* The row count is numel(Y)/N, exactly how ref_gemm.cpp gemm_nt_impl derives it. The
     * schema-required "M" parameter is NOT it: the engine passes the step-batch arena
     * capacity there (8192 at --max-num-batched-tokens 8192) while Y carries the live
     * rows ([1,N] at decode). Trusting the parameter computes a full prefill batch on
     * every decode token -- 0.35 tok/s measured -- and writes garbage into arena rows
     * the engine never reads, which no guard catches because the arena is that big. */
    M = rad_tensor_numel(Y) / N;
    if (M <= 0) return RAD_E_SHAPE;
    if (M > 0x7ffffffLL || N > 0x7ffffffLL || K > 0x7ffffffLL) return RAD_E_SHAPE;

    /* Contiguous innermost axes or nothing: this kernel's loads index elements directly. */
    if (A->rank >= 2 && A->stride[A->rank - 1] != 1) return RAD_E_STRIDE;
    if (B->rank >= 2 && B->stride[B->rank - 1] != 1) return RAD_E_STRIDE;
    if (Y->rank >= 2 && Y->stride[Y->rank - 1] != 1) return RAD_E_STRIDE;

    const int64_t a_pitch = row_stride(A, K);
    const int64_t b_pitch = row_stride(B, K);
    const int64_t y_pitch = row_stride(Y, N);
    if (rad_tensor_numel(B) < N * K) return RAD_E_SHAPE;

    hipStream_t stream = (hipStream_t)(uintptr_t)s;

    /* Decode band: M == 1 goes to the GEMV row when vectors can be trusted -- K and the
     * weight row pitch divisible by 8 (a uint4 walk stays inside the row and stays
     * aligned) and both bases 16-byte aligned. Every shape the engine records for
     * MiniCPM5 satisfies this; anything else keeps the tile kernel, which is correct
     * for all shapes, merely slow at M = 1. */
    if (M == 1 && (K % 8) == 0 && (b_pitch % 8) == 0 &&
        ((uintptr_t)A->data & 15u) == 0 && ((uintptr_t)B->data & 15u) == 0) {
        hipLaunchKernelGGL((gemv_nt_bf16_kernel), dim3((unsigned)((N + kGemvRows - 1) / kGemvRows)),
                           dim3(kGemvThreads), 0, stream,
                           (const uint16_t*)A->data, (const uint16_t*)B->data,
                           (uint16_t*)Y->data,
                           Y->dtype == RAD_F32 ? (float*)Y->data : nullptr,
                           (int)N, (int)K, b_pitch);
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }

    /* Prefill band: every M > 1 goes to the WMMA tile (gfx1100 native
     * v_wmma_f32_16x16x16_bf16, ragged edges staged to zero through LDS there). The
     * scalar tile below is the fallback for a failed m64 launch -- hipGetLastError
     * there means "never enqueued", so falling through cannot double-write Y. */
    int rc = r3d_gemm_nt_m64_launch(A->data, B->data, Y->data,
                                    Y->dtype == RAD_F32 ? (float*)Y->data : nullptr,
                                    (int)M, (int)N, (int)K, a_pitch, b_pitch, y_pitch,
                                    stream);
    if (rc == RAD_OK) return RAD_OK;

    dim3 grid((unsigned)((N + kBN - 1) / kBN), (unsigned)((M + kBM - 1) / kBM));
    hipLaunchKernelGGL((gemm_nt_bf16_kernel), grid, dim3(kThreads), 0, stream,
                       (const uint16_t*)A->data, (const uint16_t*)B->data,
                       (uint16_t*)Y->data,
                       Y->dtype == RAD_F32 ? (float*)Y->data : nullptr,
                       (int)M, (int)N, (int)K, a_pitch, b_pitch, y_pitch);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r3d_rad_gemm_nt(const RadArgs* a, RadStream s) {
    return gemm_nt_launch(a, s, "N", "K");
}

extern "C" int r3d_rad_logits_gemm(const RadArgs* a, RadStream s) {
    return gemm_nt_launch(a, s, "n_vocab", "n_embd");
}
