/* r3d_gemm_bf16_m64.hip -- the WMMA prefill tile for gemm_nt/logits_gemm on RDNA3.

 * Same contract as r3d_gemm_bf16.hip's scalar tile (Y[M,N] = A[M,K] @ B[N,K]^T, bf16):
 * operand reading follows ref_gemm.h (row pitch = the trailing-dimension fold, contiguous
 * innermost axis or the entry point refuses), every output accumulates f32 over ascending k,
 * ragged M/N/K rows load as zero and are never stored, and the single narrowing to bf16
 * happens at the store. -ffp-contract=off and the wave32 pin live in CMakeLists.
 *
 * What changes is where the multiply-adds happen: the scalar tile does 4x4 FMAs per thread
 * out of LDS, which left the XTX's matrix units idle. This tile stages the same 64x64
 * outputs per workgroup but drives them with gfx1100's native v_wmma_f32_16x16x16_bf16
 * (verified in the emitted ISA, not emulated) through rocWMMA's fragment API, so the
 * fragment lane layout is the library's problem, not ours. 256 threads = 8 wave32s, each
 * owning two horizontally adjacent 16x16 wmma tiles of the workgroup's 4x4 tile grid.
 *
 * Arithmetic drift vs libref: the scalar tile reproduced the oracle's ascending-k f32 chain
 * bit-for-bit (rel_l2 0.000e+00). A wmma op folds its 16 k-products in an opaque internal
 * order, so this tile trades the zero for a few ulps of f32 drift -- entirely below bf16
 * store rounding, exactly the reassociation class the tile comment in r3d_gemm_bf16.hip
 * sanctions, and kbench's rel_l2 gate adjudicates it. k-slices still visit ascending.
 *
 * Why LDS staging on both sides: a fragment load reads a full 16x16 with no bounds
 * predicate, so ragged edges and read red zones demand the padded-LDS staging (same rule
 * as the scalar tile); the accumulator stores go to LDS for the same reason, and the
 * guarded global copy-out below is also where f32-vs-bf16 output and the fold's y_pitch
 * live. Staging loads stay scalar-guarded on purpose: they are correct for any element
 * pitch, and at 64x32 bf16 per slice per operand they are a fraction of the tile's traffic.
 */

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "rad_abi.h"
#include "r3d_gemm.h"

/* rocWMMA's config gate whitelists specific chips and rejects family-generic targets
 * (static_assert "Unsupported architecture" on gfx11-generic), and the engine picks code
 * objects chip-first-then-family: gfx11-generic is what carries every existing libr3d row
 * onto a gfx1100 card. So the WMMA tile compiles for the RDNA3 chips only, and the generic
 * build of this TU degrades to a refusal that hands the entry back to the scalar tile in
 * r3d_gemm_bf16.hip. The standalone build therefore ships gfx1100-03 beside gfx11-generic:
 * on a 7900 the chip object wins and the WMMA path runs. */
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)

#include <rocwmma/rocwmma.hpp>

namespace {

constexpr int kMB = 64, kNB = 64, kKB = 32;   /* output tile, k slice */
constexpr int kLdsPad = 4;                    /* LDS row padding */
constexpr int kThreads = 256;                 /* 8 wave32s; 2 wmma tiles each */
constexpr int kWave = 32;                     /* wave32 ABI pinned in CMakeLists */

using bf16 = rocwmma::bfloat16_t;

__device__ __forceinline__ uint16_t r3d_f32_to_bf16_m64(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

__global__ __launch_bounds__(kThreads) void gemm_nt_bf16_m64_kernel(
    const uint16_t* __restrict__ A, const uint16_t* __restrict__ B,
    uint16_t* __restrict__ Y, float* __restrict__ Yf,
    int M, int N, int K, int64_t a_pitch, int64_t b_pitch, int64_t y_pitch) {

    __shared__ uint16_t As[kMB][kKB + kLdsPad];   /* [m][k] natural, row-major frag layout */
    __shared__ uint16_t Bs[kNB][kKB + kLdsPad];   /* [n][k]: col_major frag reads B[n][k] */
    __shared__ float    Ys[kMB][kNB + kLdsPad];   /* accumulator staging for guarded store */

    const int m0 = blockIdx.y * kMB;
    const int n0 = blockIdx.x * kNB;
    const int w = threadIdx.x / kWave;            /* wave index 0..7 */
    const int tw = w >> 1;                        /* this wave's tile row (of 4) */
    const int tc = (w & 1) * 2;                   /* its two tile columns (of 4) */

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bf16, rocwmma::row_major> fa;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bf16, rocwmma::col_major> fb0, fb1;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fc0, fc1;
    rocwmma::fill_fragment(fc0, 0.0f);
    rocwmma::fill_fragment(fc1, 0.0f);

    for (int k0 = 0; k0 < K; k0 += kKB) {
        __syncthreads();   /* previous slice's fragment reads are done in every wave */
        for (int i = threadIdx.x; i < kMB * kKB; i += kThreads) {
            const int r = i / kKB, c = i % kKB;
            const int m = m0 + r, k = k0 + c;
            As[r][c] = (m < M && k < K) ? A[(int64_t)m * a_pitch + k] : (uint16_t)0;
        }
        for (int i = threadIdx.x; i < kNB * kKB; i += kThreads) {
            const int r = i / kKB, c = i % kKB;
            const int n = n0 + r, k = k0 + c;
            Bs[r][c] = (n < N && k < K) ? B[(int64_t)n * b_pitch + k] : (uint16_t)0;
        }
        __syncthreads();

        /* B^T's [k][n] tile is col_major at &Bs[n][0]: element (k, n) = ptr[n*ld + k],
         * which is exactly Bs[n][k] with ld = the padded row stride. */
        rocwmma::load_matrix_sync(fa,  reinterpret_cast<const bf16*>(&As[tw * 16][0]),
                                  kKB + kLdsPad, rocwmma::mem_row_major);
        rocwmma::load_matrix_sync(fb0, reinterpret_cast<const bf16*>(&Bs[tc * 16][0]),
                                  kKB + kLdsPad);
        rocwmma::load_matrix_sync(fb1, reinterpret_cast<const bf16*>(&Bs[(tc + 1) * 16][0]),
                                  kKB + kLdsPad);
        rocwmma::mma_sync(fc0, fa, fb0, fc0);
        rocwmma::mma_sync(fc1, fa, fb1, fc1);
    }

    __syncthreads();   /* every wave's fragment LDS reads finished before Ys is written */
    rocwmma::store_matrix_sync(&Ys[tw * 16][tc * 16], fc0, kNB + kLdsPad,
                               rocwmma::mem_row_major);
    rocwmma::store_matrix_sync(&Ys[tw * 16][(tc + 1) * 16], fc1, kNB + kLdsPad,
                               rocwmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < kMB * kNB; i += kThreads) {
        const int r = i / kNB, c = i % kNB;
        const int m = m0 + r, n = n0 + c;
        if (m < M && n < N) {
            const int64_t yb = (int64_t)m * y_pitch;
            if (Yf) Yf[yb + n] = Ys[r][c];
            else    Y[yb + n]  = r3d_f32_to_bf16_m64(Ys[r][c]);
        }
    }
}

} /* anonymous namespace */

int r3d_gemm_nt_m64_launch(const void* A, const void* B, void* Y, float* Yf,
                           int M, int N, int K, int64_t a_pitch, int64_t b_pitch,
                           int64_t y_pitch, hipStream_t stream) {
    dim3 grid((unsigned)((N + kNB - 1) / kNB), (unsigned)((M + kMB - 1) / kMB));
    hipLaunchKernelGGL((gemm_nt_bf16_m64_kernel), grid, dim3(kThreads), 0, stream,
                       (const uint16_t*)A, (const uint16_t*)B, (uint16_t*)Y, Yf,
                       M, N, K, a_pitch, b_pitch, y_pitch);
    const hipError_t le = hipGetLastError();
    if (le != hipSuccess) fprintf(stderr, "libr3d: m64 launch FAILED (%s)\n",
                                 hipGetErrorName(le));
    else if (getenv("R3D_DEBUG_GEMM")) {
        static bool once = false;
        if (!once) { once = true; fprintf(stderr, "libr3d: m64 WMMA tile launched\n"); }
    }
    return le == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

#else  /* gfx11-generic and every other target: refuse, scalar tile takes the op */

int r3d_gemm_nt_m64_launch(const void*, const void*, void*, float*,
                           int, int, int, int64_t, int64_t, int64_t, hipStream_t) {
    static bool once = false;
    if (!once && getenv("R3D_DEBUG_GEMM")) {
        once = true;
        fprintf(stderr, "libr3d: m64 refusal stub ran (generic code object selected)\n");
    }
    return RAD_E_DEVICE;
}

#endif
