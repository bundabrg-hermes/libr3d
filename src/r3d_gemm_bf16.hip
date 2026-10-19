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
 * This is the correctness-first tile: 64x64 outputs per workgroup, K in 16-slice LDS tiles,
 * 256 threads at 4x4 outputs each, scalar bf16 loads, no WMMA yet. Decode (M=1) therefore
 * wastes 63/64 of every A tile; the m64/WMMA variants libr4d runs on RDNA4 are the speed
 * roadmap. What matters at this commit is that every recorded geometry comes out RIGHT.
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
extern "C" int r3d_rad_gemm_nt(const RadArgs* a, RadStream s) {
    const RadTensor* A = rad_arg_in(a, 0);
    const RadTensor* B = rad_arg_in(a, 1);
    const RadTensor* Y = rad_arg_in(a, 2);
    if (!A || !B || !Y) return RAD_E_INVAL;
    if (A->dtype != RAD_BF16 || B->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (Y->dtype != RAD_BF16 && Y->dtype != RAD_F32) return RAD_E_DTYPE;

    long long M = 0, N = 0, K = 0;
    if (!rad_args_geti(a, "M", &M) || !rad_args_geti(a, "N", &N) || !rad_args_geti(a, "K", &K))
        return RAD_E_SHAPE;
    if (M <= 0 || N <= 0 || K <= 0) return RAD_E_SHAPE;
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
    dim3 grid((unsigned)((N + kBN - 1) / kBN), (unsigned)((M + kBM - 1) / kBM));
    hipLaunchKernelGGL((gemm_nt_bf16_kernel), grid, dim3(kThreads), 0, stream,
                       (const uint16_t*)A->data, (const uint16_t*)B->data,
                       (uint16_t*)Y->data,
                       Y->dtype == RAD_F32 ? (float*)Y->data : nullptr,
                       (int)M, (int)N, (int)K, a_pitch, b_pitch, y_pitch);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
