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
 *
 * Two schedules share this kernel via the DBUF template parameter. The shipped default
 * stages slice, barriers, computes, barriers, restages -- every slice eats the full
 * global-load latency before WMMA starts. DBUF=1 pipelines it the way llama-cpp-rdna-boosts
 * block 06 (MMB, gfx1100-validated) does: next slice prefetched global->registers while
 * this slice's WMMA runs, A staging double-buffered so the next stage overlaps compute,
 * B deliberately NOT (block 06 measured B's second buffer as a loss). Selected per launch
 * by R3D_GEMM_NO_DBUF=1 (kill switch); the DBUF schedule is the promoted default.
 */

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rad_abi.h"
#include "r3d_gemm.h"

/* rocWMMA's config gate whitelists specific chips and rejects family-generic targets
 * (static_assert "Unsupported architecture" on gfx11-generic), so the WMMA kernel body
 * compiles for the RDNA3 chips only. The LAUNCHER cannot live behind the same #if: an
 * unmarked function is __host__ only, and the host pass sees no __gfxNNNN__ macro at all,
 * so gating the launcher itself compiled the refusal into the host object on every target
 * and the WMMA tile was unreachable even where it was built. The launcher therefore lives
 * outside the gate and decides at runtime by the device's gcnArchName -- the same list the
 * engine would have matched chip-first. On a chip the runtime does not have a gfxNNNN code
 * object for, the launch fails with hipErrorNotFound and the caller falls back to the
 * scalar tile exactly as before. */
constexpr int kMB = 64, kNB = 64, kKB = 16;   /* output tile, k slice */
/* kKB MUST equal the fragment kDim (16). A 16x16x16 fragment consumes exactly
 * 16 k columns per load+mma pair; staging a 32-wide slice while the fragments
 * read only columns 0..15 silently dropped half of every operand's k range
 * (rel_l2 ~ 0.5 everywhere, bit-exact only when K <= 16). If widening kKB for
 * double-buffering, drive two fragment loads per slice at +0/+16 k offsets. */
constexpr int kLdsPad = 4;                    /* LDS row padding */
constexpr int kThreads = 256;                 /* 8 wave32s; 2 wmma tiles each */
constexpr int kWave = 32;                     /* wave32 ABI pinned in CMakeLists */

/* Staging per thread per slice: 64 rows x 16 cols / 256 threads = 4 elements.
 * Thread tid covers column c = tid % 16 of rows tid/16 + 16j (j = 0..3): a warp
 * reads 2 rows x 16 contiguous columns per instruction, same coalescing as the
 * strided loop the single-buffer path uses. */
constexpr int kItems = (kMB * kKB) / kThreads;

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)

#include <rocwmma/rocwmma.hpp>

namespace {

using bf16 = rocwmma::bfloat16_t;

__device__ __forceinline__ uint16_t r3d_f32_to_bf16_m64(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* DBUF selects the k-slice pipeline (structure borrowed from llama-cpp-rdna-boosts
 * block 06 MMB, gfx1100-validated): with DBUF the A staging buffer is doubled, so
 * the NEXT slice's stage writes the other buffer and overlaps this slice's WMMA
 * instead of serialising behind the stage->barrier->compute chain. Operand global
 * loads are issued to registers one slice ahead (prefetch), so the post-barrier
 * stage reads registers only.
 *
 * B stays single-buffered deliberately. Block 06 measured B's second buffer as a
 * loss (their GLU 1225 -> 1433 ms): the extra LDS traffic costs more than the
 * removed barrier buys. The single barrier before Bs is re-staged stays.
 *
 * Barrier proof for the DBUF schedule (per slice ks, buffers ab = ks&1):
 *   prefetch(ks+1)          global -> regs; completes before the stage that reads them
 *   stage_a(ks+1 -> !ab)    disjoint from the frag reads of ab this iteration
 *   frag loads + mma        read As[ab] and Bs
 *   sync                    every wave's frag reads of As[ab]/Bs are done
 *   stage_b(ks+1 -> Bs)     now safe; Bs is dead from ks's frag loads
 *   sync                    stage_b visible for ks+1; also orders stage_a(ks+1),
 *                           whose write precedes this barrier in every thread's
 *                           program order, so ks+1's frag loads of As[!ab] see it.
 */
template <bool DBUF>
__global__ __launch_bounds__(kThreads) void gemm_nt_bf16_m64_kernel(
    const uint16_t* __restrict__ A, const uint16_t* __restrict__ B,
    uint16_t* __restrict__ Y, float* __restrict__ Yf,
    int M, int N, int K, int64_t a_pitch, int64_t b_pitch, int64_t y_pitch) {

    __shared__ uint16_t As[DBUF ? 2 : 1][kMB][kKB + kLdsPad];  /* [m][k] natural, row-major frag layout */
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

    /* Raw-bit staging of one slice into As[ab]: never construct a bf16 from a
     * loaded uint16 (that CONVERTS; see the 6c7294e postmortem). */
    const int c_col = threadIdx.x % kKB;          /* staging mapping (see kItems) */
    const int r_row = threadIdx.x / kKB;

    uint16_t areg[kItems], breg[kItems];
    auto prefetch = [&](int k0) {
        #pragma unroll
        for (int j = 0; j < kItems; ++j) {
            const int i = threadIdx.x + j * kThreads;
            const int ra = i / kKB, ka = k0 + i % kKB;   /* == r_row + 16j, c_col */
            const int ma = m0 + ra, n = n0 + i / kKB, kb = ka;
            areg[j] = (ma < M && ka < K) ? A[(int64_t)ma * a_pitch + ka] : (uint16_t)0;
            breg[j] = (n < N && kb < K)  ? B[(int64_t)n  * b_pitch + kb] : (uint16_t)0;
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

    const int nks = (K + kKB - 1) / kKB;
    if (DBUF) {
        prefetch(0);
        stage_a(0); stage_b();
        __syncthreads();                          /* prologue slices visible */
        for (int ks = 0; ks < nks; ++ks) {
            if (ks + 1 < nks) prefetch((ks + 1) * kKB);
            if (ks + 1 < nks) stage_a((ks + 1) & 1);   /* other A buffer: overlaps WMMA */

            const bf16* ap = reinterpret_cast<const bf16*>(&As[ks & 1][tw * 16][0]);
            const bf16* bp0 = reinterpret_cast<const bf16*>(&Bs[tc * 16][0]);
            const bf16* bp1 = reinterpret_cast<const bf16*>(&Bs[(tc + 1) * 16][0]);
            /* B^T's [k][n] tile is col_major at &Bs[n][0]: element (k, n) = ptr[n*ld + k],
             * which is exactly Bs[n][k] with ld = the padded row stride. */
            rocwmma::load_matrix_sync(fa,  ap,  kKB + kLdsPad, rocwmma::mem_row_major);
            rocwmma::load_matrix_sync(fb0, bp0, kKB + kLdsPad);
            rocwmma::load_matrix_sync(fb1, bp1, kKB + kLdsPad);
            rocwmma::mma_sync(fc0, fa, fb0, fc0);
            rocwmma::mma_sync(fc1, fa, fb1, fc1);

            __syncthreads();                      /* As[ab]/Bs reads finished */
            if (ks + 1 < nks) stage_b();          /* Bs single-buffered: after the barrier */
            __syncthreads();                      /* next slices visible */
        }
    } else {
        for (int k0 = 0; k0 < K; k0 += kKB) {
            __syncthreads();   /* previous slice's fragment reads are done in every wave */
            for (int i = threadIdx.x; i < kMB * kKB; i += kThreads) {
                const int r = i / kKB, c = i % kKB;
                const int m = m0 + r, k = k0 + c;
                As[0][r][c] = (m < M && k < K) ? A[(int64_t)m * a_pitch + k] : (uint16_t)0;
            }
            for (int i = threadIdx.x; i < kNB * kKB; i += kThreads) {
                const int r = i / kKB, c = i % kKB;
                const int n = n0 + r, k = k0 + c;
                Bs[r][c] = (n < N && k < K) ? B[(int64_t)n * b_pitch + k] : (uint16_t)0;
            }
            __syncthreads();

            /* B^T's [k][n] tile is col_major at &Bs[n][0]: element (k, n) = ptr[n*ld + k],
             * which is exactly Bs[n][k] with ld = the padded row stride. */
            rocwmma::load_matrix_sync(fa,  reinterpret_cast<const bf16*>(&As[0][tw * 16][0]),
                                      kKB + kLdsPad, rocwmma::mem_row_major);
            rocwmma::load_matrix_sync(fb0, reinterpret_cast<const bf16*>(&Bs[tc * 16][0]),
                                      kKB + kLdsPad);
            rocwmma::load_matrix_sync(fb1, reinterpret_cast<const bf16*>(&Bs[(tc + 1) * 16][0]),
                                      kKB + kLdsPad);
            rocwmma::mma_sync(fc0, fa, fb0, fc0);
            rocwmma::mma_sync(fc1, fa, fb1, fc1);
        }
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

/* ---------------------------------------------------------------------------
 * Q1: vector-staged sibling of the shipped tile (docs/REVIEW_QUEUE.md Q1:
 * QFN-P1 + Astra-P3 convergence + the Q17-delta1 donor fold-ins).
 *
 * The shipped tile stages each 64x16 slice with per-thread uint16 loads. This
 * arm ports the attention body's shipped vec pattern (ld_kv_regs/st_kv_regs in
 * r3d_attn_wmma_body.inc): 256 threads split by operand -- threads 0..127 stage
 * A, threads 128..255 stage B, and each thread moves one uint4 (8 bf16 = a 32B
 * contiguous wave segment per row pair). The operand split is wave-aligned
 * (waves 0-3 only ever touch A, waves 4-7 only B), so there is no intra-wave
 * divergence on the operand branch.
 *
 * BIT-IDENTITY BY SLOT BIJECTION: thread s = tid&127 names LDS slot (row s>>1,
 * 8-column group s&1); every slot of every slice gets exactly one writer, the
 * same global bytes land in the same slot as the scalar walk, and the fragment
 * loads consume the identical slice contents in the identical order. The only
 * permitted divergence is WHERE a thread's bytes come from in issue order, via
 * the ROT fold below -- and rotation maps slot->slot by the same permutation it
 * applies to global rows, so contents stay exact. Ragged-K rows fill the slot
 * with guarded scalars (attention's odd-base fallback shape); ragged rows write
 * zeros; both match the scalar walk's zero-fill contract exactly.
 *
 * kLdsPad CANNOT be reused: a uint4 LDS store needs the row stride 16B-aligned,
 * so this kernel carries kVecPad = 16 (stride 32 elements = 64B; the shipped
 * stride 20 is not). Consequence: LDS/WG grows 25 -> 38 KB, one workgroup per
 * CU instead of two -- exactly the occupancy trade the card A/B adjudicates.
 * (Padding also spreads fragment-load bank phases; the donor measured 16 and
 * attention's Carve strides are vec-friendly for the same reason.)
 *
 * Donor fold-ins (Q17 delta 1, operator-released), both host-gated in the
 * launcher and both off by default in this arm build:
 *   R3D_M64_VEC -- vec staging, gated to MT>=2 row tiles: the donor measured
 *       3-10% LOSES at one row tile, so the single-tile regime keeps the scalar
 *       walk. Needs 16B bases + pitch%8==0 (else the "vec" load walks a row
 *       that can straddle the buffer end -- fall back to the scalar arm).
 *   R3D_M64_ROT -- per-(tile, k-slice) issue rotation (blockIdx.x + blockIdx.y
 *       + ks) % 32, gated to K % 4096 == 0: the donor's 60-vs-185 GB/s
 *       channel-camp pair came from K%4096==0 streams, and rotation costs a few
 *       % at odd K. Skipped donor variants (recorded so nobody retries them):
 *       64B windows, 16-step groups, register-held groups across a start switch.
 * DBUF only: the single-buffer path is the kill switch and stays the shipped
 * scalar walk verbatim.
 * ------------------------------------------------------------------------- */
constexpr int kVecPad = 16;

template <bool ROT>
__global__ __launch_bounds__(kThreads) void gemm_nt_bf16_m64_vec_kernel(
    const uint16_t* __restrict__ A, const uint16_t* __restrict__ B,
    uint16_t* __restrict__ Y, float* __restrict__ Yf,
    int M, int N, int K, int64_t a_pitch, int64_t b_pitch, int64_t y_pitch) {

    constexpr int kLD = kKB + kVecPad;               /* 32 elements: 16B-aligned rows */
    static_assert(kLD * 2 % 16 == 0, "vec LDS stores need 16B-aligned rows");
    __shared__ uint16_t As[2][kMB][kLD];             /* DBUF only: always doubled */
    __shared__ uint16_t Bs[kNB][kLD];                /* B single-buffered (block-06 law) */
    __shared__ float    Ys[kMB][kNB + kVecPad];

    const int m0 = blockIdx.y * kMB;
    const int n0 = blockIdx.x * kNB;
    const int w = threadIdx.x / kWave;
    const int tw = w >> 1;
    const int tc = (w & 1) * 2;

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bf16, rocwmma::row_major> fa;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bf16, rocwmma::col_major> fb0, fb1;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fc0, fc1;
    rocwmma::fill_fragment(fc0, 0.0f);
    rocwmma::fill_fragment(fc1, 0.0f);

    const int srow = (threadIdx.x & 127) >> 1;       /* this thread's tile row 0..63 */
    const int d8   = (threadIdx.x & 1) << 3;         /* its 8-column group: 0 or 8 */
    const bool is_b = threadIdx.x >= 128;

    /* Slot index after the (optional) issue rotation: a permutation of rows, so
     * every slot keeps exactly one writer. */
    auto slot = [&](int ks) {
        return ROT ? (srow + (int)(((unsigned)(blockIdx.x + blockIdx.y + ks)) % kWave)) & (kMB - 1)
                   : srow;
    };

    uint4 areg, breg;
    auto ld_vec = [&](const uint16_t* __restrict__ Op, int64_t pitch, int base, int lim,
                      int k0, int r, uint4* dst) {
        const int gr = base + r;
        if (gr < lim && k0 + d8 + 8 <= K) {
            *dst = *reinterpret_cast<const uint4*>(Op + (int64_t)gr * pitch + k0 + d8);
        } else if (gr < lim) {                       /* ragged K tail: guarded scalars */
            uint16_t t8[8];
            #pragma unroll
            for (int u = 0; u < 8; ++u)
                t8[u] = (k0 + d8 + u < K) ? Op[(int64_t)gr * pitch + k0 + d8 + u]
                                          : (uint16_t)0;
            *dst = *reinterpret_cast<const uint4*>(t8);
        } else *dst = make_uint4(0u, 0u, 0u, 0u);    /* ragged rows: zero-fill */
    };
    auto prefetch = [&](int k0, int rot) {
        if (is_b) ld_vec(B, b_pitch, n0, N, k0, rot, &breg);
        else      ld_vec(A, a_pitch, m0, M, k0, rot, &areg);
    };
    auto stage_a = [&](int ab, int rot) {
        if (!is_b) *reinterpret_cast<uint4*>(&As[ab][rot][d8]) = areg;
    };
    auto stage_b = [&](int rot) {
        if (is_b) *reinterpret_cast<uint4*>(&Bs[rot][d8]) = breg;
    };

    const int nks = (K + kKB - 1) / kKB;
    prefetch(0, slot(0));
    stage_a(0, slot(0)); stage_b(slot(0));
    __syncthreads();                                 /* prologue slices visible */
    for (int ks = 0; ks < nks; ++ks) {
        if (ks + 1 < nks) { prefetch((ks + 1) * kKB, slot(ks + 1));
                            stage_a((ks + 1) & 1, slot(ks + 1)); }

        const bf16* ap = reinterpret_cast<const bf16*>(&As[ks & 1][tw * 16][0]);
        const bf16* bp0 = reinterpret_cast<const bf16*>(&Bs[tc * 16][0]);
        const bf16* bp1 = reinterpret_cast<const bf16*>(&Bs[(tc + 1) * 16][0]);
        rocwmma::load_matrix_sync(fa,  ap,  kLD, rocwmma::mem_row_major);
        rocwmma::load_matrix_sync(fb0, bp0, kLD);
        rocwmma::load_matrix_sync(fb1, bp1, kLD);
        rocwmma::mma_sync(fc0, fa, fb0, fc0);
        rocwmma::mma_sync(fc1, fa, fb1, fc1);

        __syncthreads();                             /* As[ab]/Bs reads finished */
        if (ks + 1 < nks) stage_b(slot(ks + 1));
        __syncthreads();                             /* next slices visible */
    }

    __syncthreads();
    rocwmma::store_matrix_sync(&Ys[tw * 16][tc * 16], fc0, kNB + kVecPad,
                               rocwmma::mem_row_major);
    rocwmma::store_matrix_sync(&Ys[tw * 16][(tc + 1) * 16], fc1, kNB + kVecPad,
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

#else  /* generic and every non-RDNA3 target: the kernel exists as a no-op so the host
        * pass has the symbol to launch; the launcher below never dispatches it there. */

namespace {
template <bool DBUF>
__global__ __launch_bounds__(kThreads) void gemm_nt_bf16_m64_kernel(
    const uint16_t*, const uint16_t*, uint16_t*, float*,
    int, int, int, int64_t, int64_t, int64_t) {}
template <bool ROT>
__global__ __launch_bounds__(kThreads) void gemm_nt_bf16_m64_vec_kernel(
    const uint16_t*, const uint16_t*, uint16_t*, float*,
    int, int, int, int64_t, int64_t, int64_t) {}
} /* anonymous namespace */

#endif

/* Variant selection at launch, both instantiated in the same TU:
 *   default          -> the DBUF k-slice pipeline (promoted 2026-10-20 after the card
 *                       A/B: serve TTFT 0.356 -> 0.289 s median @981 and 0.691 -> 0.550
 *                       s @1961, ~1.25x, non-overlapping distributions; kbench gemm_nt
 *                       269/269 with identical worst rel_l2 both arms)
 *   R3D_GEMM_NO_DBUF=1 -> the original single-buffer schedule, kill switch for
 *                       measurement and rollback
 * The two schedules produce bit-identical tiles (same slice contents, same ascending-k
 * fragment order -- only the LDS write timing moves), so the distribution gate is
 * implied by the schedule being arithmetic-preserving; kbench's identical worst rel_l2
 * corroborates. */
int r3d_gemm_nt_m64_launch(const void* A, const void* B, void* Y, float* Yf,
                           int M, int N, int K, int64_t a_pitch, int64_t b_pitch,
                           int64_t y_pitch, hipStream_t stream) {
    /* Chip gate, evaluated once against the device this stream's launches land on. */
    static int allowed = -1;
    if (allowed < 0) {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess) dev = 0;
        hipDeviceProp_t p{};
        allowed = (hipGetDeviceProperties(&p, dev) == hipSuccess) &&
                  (strstr(p.gcnArchName, "gfx1100") || strstr(p.gcnArchName, "gfx1101") ||
                   strstr(p.gcnArchName, "gfx1102") || strstr(p.gcnArchName, "gfx1103"));
    }
    if (!allowed) {
        static bool once = false;
        if (!once && getenv("R3D_DEBUG_GEMM")) {
            once = true;
            fprintf(stderr, "libr3d: m64 refused (device is not an RDNA3 chip)\n");
        }
        return RAD_E_DEVICE;
    }
    static const bool dbuf = getenv("R3D_GEMM_NO_DBUF") == nullptr;
    /* Q1 arms, both dark until released: vec staging (MT>=2 rows, aligned
     * operands, DBUF schedule) and the K%4096==0-gated issue rotation on top.
     * Withdraw law (issue #6/Q6): the literal "0" is OFF, any other set value
     * is ON. */
    static const bool vec  = getenv("R3D_M64_VEC")  && strcmp(getenv("R3D_M64_VEC"),  "0") != 0;
    static const bool rot  = getenv("R3D_M64_ROT")  && strcmp(getenv("R3D_M64_ROT"),  "0") != 0;
    dim3 grid((unsigned)((N + kNB - 1) / kNB), (unsigned)((M + kMB - 1) / kMB));
    const int MT = (M + kMB - 1) / kMB;
    const bool vec_ok = vec && dbuf && MT >= 2 &&
                        (a_pitch % 8 == 0) && (b_pitch % 8 == 0) &&
                        ((((uintptr_t)A) & 15) == 0) && ((((uintptr_t)B) & 15) == 0);
    const bool rot_on = vec_ok && rot && (K % 4096 == 0);
    if (vec_ok && rot_on)
        hipLaunchKernelGGL((gemm_nt_bf16_m64_vec_kernel<true>), grid, dim3(kThreads), 0, stream,
                           (const uint16_t*)A, (const uint16_t*)B, (uint16_t*)Y, Yf,
                           M, N, K, a_pitch, b_pitch, y_pitch);
    else if (vec_ok)
        hipLaunchKernelGGL((gemm_nt_bf16_m64_vec_kernel<false>), grid, dim3(kThreads), 0, stream,
                           (const uint16_t*)A, (const uint16_t*)B, (uint16_t*)Y, Yf,
                           M, N, K, a_pitch, b_pitch, y_pitch);
    else if (dbuf)
        hipLaunchKernelGGL((gemm_nt_bf16_m64_kernel<true>), grid, dim3(kThreads), 0, stream,
                           (const uint16_t*)A, (const uint16_t*)B, (uint16_t*)Y, Yf,
                           M, N, K, a_pitch, b_pitch, y_pitch);
    else
        hipLaunchKernelGGL((gemm_nt_bf16_m64_kernel<false>), grid, dim3(kThreads), 0, stream,
                           (const uint16_t*)A, (const uint16_t*)B, (uint16_t*)Y, Yf,
                           M, N, K, a_pitch, b_pitch, y_pitch);
    const hipError_t le = hipGetLastError();
    if (le != hipSuccess) fprintf(stderr, "libr3d: m64 launch FAILED (%s)\n",
                                 hipGetErrorName(le));
    else if (getenv("R3D_DEBUG_GEMM")) {
        static bool once = false;
        if (!once) { once = true; fprintf(stderr, "libr3d: m64 WMMA tile launched (%s)\n",
                                          rot_on ? "vec staging rot=on"
                                       : vec_ok ? "vec staging rot=off"
                                       : dbuf   ? "dbuf pipeline" : "single-buffer"); }
    }
    return le == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
