/* r3d_gemm_w2a8.hip -- gemm_nt_q at the `w2a8` encoding on RDNA3: the DFlash2 draft head.
 *
 * DRAFT for libr3d (issue: w2a8 decode band). Do not treat as shipped until the
 * registration snippets at the tail of this file are applied to r3d_rows.cpp /
 * CMakeLists.txt and the kbench gate in tools/gate.sh runs on a w2a8 fixture.
 *
 * The band (v1.11 arch/common/rad_block_dflash2.h, lines ~397-415 and ~838-843):
 * when `dflash.draft_head.weight` carries its 2-bit recipe, the drafter's coarse lm_head
 * declares
 *
 *   gemm_nt_q(M in 1..srows, N = head_rows, K = n_embd, group = 128, dtype "w2a8")
 *     weights  b       = the "codes" plane of the draft head, u2 dense, [N, K] elements
 *                        (the head is its own checkpoint tensor, mapped from the target's
 *                        lm_head by map_copy -- NOT output.weight)
 *              b_scale = the "scale,zero" view: an f16 scale and an integer zero per
 *                        (row, 128 of K), affine, two planes of one tensor
 *     inputs   a       = dq  [M, K]      i8   quant_act_i8's codes, IN STEP ORDER
 *              a_scale = ds  [M, 1]      f32  one scale a row
 *              a_sum   = dsum[K/128, pad] f32 quant_act_i8's group sums, GROUP-major
 *     output   y       = dlog [M, N]     bf16
 *
 * which is libr4d's `gemm_w2a8_nt` row (r4d_rows.cpp ~7485, constraint table cGemmNtW2A8
 * ~4495) and libref's `gemm_zp` ZP_I8_ROW arm (ref_gemm.cpp ~344-375, grid table
 * ref_planes.h:118 { "w2a8", RAD_U2, ZP_I8_ROW }).
 *
 * ---- the oracle's arithmetic, transcribed exactly (ref_gemm.cpp gemm_zp) ----
 *
 *   G    = K / 128;  mpad = ceil(M/16)*16;  codes q[k] = the RAW u2 value 0..3
 *                    (rad_load_bits LSB-first: q[k] = (crow[k>>2] >> (2*(k&3))) & 3 --
 *                    there is NO offset-2 signed form; the zero point carries the offset)
 *   A[k] = a[i8_step_pos(k)]  per row, i8_step_pos swaps bits 2..3 of k (ref_common.h:580,
 *                    its own inverse; quant_act_i8 stores columns in that order)
 *   for each output (m, n):
 *     acc = 0.0f                                                  (f32)
 *     for each group g in ascending order:
 *       is = SUM_{k in g} (int64) A[m][k] * q[n][k]               exact integer
 *       sub = (float)is - ((float)z[n][g] * (float)a_sum[g*mpad + m])   two f32 ops
 *       acc += (sub * sa[m]) * sb[n][g]                           two RN multiplies,
 *                                                                 scale-then-scale,
 *                                                                 then an RN add
 *     y[m][n] = bf16(acc)                                         one RNE narrowing
 *
 * The kernel is bit-exact to this, not merely close: every value up to the scales is an
 * exact f32 integer -- |is| <= 128*127*3 = 48768, |z| <= 255, |a_sum| <= 128*127 = 16256,
 * |z * a_sum| <= 4145280, and every partial sum of any fold order stays an integer below
 * 2^24 -- so the octet butterfly may reassociate the k chain while `is` is still integers,
 * and the flush from `sub` onward replays the oracle's op order op for op. (The fp8a8
 * house line -- rel_l2, kbench adjudicates -- is not even needed here; it IS needed
 * nowhere, but the gate decides either way.)
 *
 * ---- the kernel: a decode-band GEMV, the r3d_gemm_fp8a8.hip skeleton at 2 bits ----
 *
 * N = 248320 vocab rows x K = 5120 makes this the bandwidth band the fp8a8 decode row is:
 * warp per weight row (one workgroup of 256 threads covers eight), and where the fp8 row
 * takes one aligned dword per lane per 128-of-K group, the u2 codes take one aligned dword
 * per lane per FOUR groups -- sixteen ascending codes per lane, 32 lanes x 4 B = one
 * 128-byte wavefront per 512 codes, every weight byte read exactly once. Lanes 8j..8j+7
 * cover group g0+j (a lane's 16 codes never straddle a group: 16*(l mod 8) + 16 <= 128),
 * so the group partial folds through an OCTET butterfly (xor 4,2,1 -- exact, see above)
 * and broadcasts to the wave; every lane then flushes the window's four groups in
 * ascending order, holding its own copy of the acc chain -- the fp8 row's redundant-acc
 * structure, which keeps the across-group order identical to the oracle's.
 *
 * The activation is staged per 512-code window through LDS as f32 (`xs`, MT x 512 floats,
 * 16 KB at MT=8 -- the fp8 row's xs pattern with the step-order de-permute folded into the
 * staging index and a permutation that makes both the staging write and the fold's
 * float4 reads conflict-free). The `a_sum` correction, `a_scale` (per ROW, not per group:
 * quant_act_i8's grid) and the (scale, zero) of the group all ride the flush in the
 * oracle's order. K % 128 == 0 is the row gate, so tail GROUPS are whole groups: the
 * groups past the last whole dword window take a byte-per-lane walk (all 32 lanes cover
 * one group, full-wave butterfly); their weight bytes are in-row and staging bounds match
 * ref's "zero past K, never a stray decode" rule for the padded MT rows.
 *
 * ---- the weight layout: rad_planes.h's canonical joined form, not libr4d's fragments ----
 *
 * b (operand 2, the codes plane) is stored as it is -- the layout hook declines it with
 * RAD_E_UNSUPPORTED, exactly as ref_layout_gemm_q does (ref_planes.h:192: "The codes
 * (operand 2) are one plane and stored as they are"). b_scale (operand 3) arrives as the
 * two planes "scale","zero"; the hooks join them with rad_planes_layout -- the same bytes
 * under the same tag "rad.planes" that libref's ref_layout_gemm_q answers with, so the
 * oracle reads the stored weight the same way the kernel does. libr4d instead invents a
 * fragment order (R4D_DT_W2_FRAG) and a packed sz dword (R4D_DT_W4_SZ, f16 scale plus
 * f16(-(1024+zero))) for its wave-wide WMMA walk; this GEMV wants neither -- it walks the
 * codes row-major and reads the two planes directly through the joined header, and one
 * fewer private format is one fewer thing that can disagree. (libr4d's rows are RDNA4's;
 * on gfx11 libr3d is the only device library, so no co-loader can ask for those bytes.)
 *
 * Deviations from ref/r4d, each a DECLINE so the band resolves to the host rather than a
 * wrong number: any encoding transform (ref also accepts the "givens" activation rotation;
 * this kernel does not rotate, so the layout hook refuses at DECLARE time -- an issue-time
 * refusal after a declare-time accept is exactly the abort fault the R3D_NO_FP8A8 story
 * documents). A zero plane wider than u4 or narrower than u4 (u8 and u4 are served, the
 * checkpoint's u8 included; ref's zero_dt also accepts u2, r4d also accepts u4). A scale
 * plane that is not f16. pb1/pb2 parts (ref's parts path is e2m1-only; gemm_zp refuses
 * parts at ZP_I8_ROW outright). b_ref (ref accepts it beside an e2m1 weight only). No
 * a_sum (the grid's zero point needs it; ref refuses the launch without it).
 *
 * M derives from numel(y)/N -- NEVER the schema's M parameter (the arena-capacity rule).
 *
 * -ffp-contract=off is plugin-wide via rad_add_plugin; the -mno-wavefrontsize64 pin is a
 * CMake line in the tail snippets. R3D_NO_DFLASH=1 withdraws the row at the registration
 * gate (the drafter family's lever); the entry keeps a backstop refusal only.
 */

#include "rad_abi.h"
#include "rad_plugin.h"   /* rad_arg_in / rad_tensor_numel / rad_param_get* (host helpers) */
#include "rad_planes.h"
#include "rad_encoding.h"

#include <cstring>
#include <cstdlib>
#include <cstdio>

#ifndef R3D_W2A8_HOST_GATE   /* tools/gate_w2a8.cpp includes ONLY the host parts */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "r3d_dump.h"

namespace {

/* i8_step_pos, ref_common.h:580 verbatim (bits 2..3 swapped; its own inverse). Column k of
 * quant_act_i8's row lives at index i8_step_pos(k); the fold reads columns ascending, so
 * the de-permute rides the staging index. */
__device__ __forceinline__ int w2_step_pos(int k) {
    return (k & ~12) | ((k & 4) << 1) | ((k & 8) >> 1);
}

/* ref_gemm.h row_stride, verbatim (the fp8a8 TU's f8_ twin): fold trailing dims until
 * they span `cols`; the next axis out carries the row. */
int64_t w2_row_stride(const RadTensor* t, int64_t cols) {
    if (!t || t->rank < 2) return cols;
    int64_t span = 1;
    for (int i = (int)t->rank - 1; i >= 0; --i) {
        span *= t->shape[i];
        if (span == cols) return (i > 0) ? t->stride[i - 1] : cols;
        if (span > cols) break;
    }
    return t->stride[t->rank - 2];
}

constexpr int kThreads = 256;
constexpr int kWave    = 32;

}  /* anonymous namespace */

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)

namespace {

/* ref_common.h's bf16 narrowing (the fp8a8/logits TUs' shared form): RTNE, NaN -> quiet. */
__device__ __forceinline__ uint16_t w2_f32_to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* The xs permutation: element kk of a 512-code window of row m lives at
 *     j*128 + u*32 + o*4 + e   for   kk = j*128 + o*16 + u*4 + e.
 * The staging loop's four consecutive writes land in consecutive slots (float4 writes,
 * thread-t contiguous), and in the fold the lanes of one octet read eight CONSECUTIVE
 * float4s -- one conflict-free 128 B LDS phase per eight lanes, wave32. */
__device__ __forceinline__ int w2_xs_idx(int j, int kk) {   /* j = kk >> 7 (0..3) */
    return (j << 7) + (((kk >> 2) & 3) << 5) + (((kk >> 4) & 7) << 2) + (kk & 3);
}

/* MT activation rows per block; W = prefetch window in dwords (a dword covers FOUR scale
 * groups -- the fp8 row's PF law, issue #7, with the window counted in load units). */
template <int MT, int W>
__global__ __launch_bounds__(kThreads, 2) void gemm_nt_w2a8_gemv_kernel(
    const int8_t* __restrict__ A,       /* [M, a_pitch] i8 codes, step order */
    const uint8_t* __restrict__ B,      /* [N, b_pitch] u2 codes, dense (stored as is) */
    uint16_t* __restrict__ Y,           /* [M, y_pitch] bf16 */
    const float* __restrict__ AS,       /* AS[m * as_pitch]                    f32 */
    const __half* __restrict__ SC,      /* [N, sc_pitch] f16  (scale plane) */
    const uint8_t* __restrict__ ZP,     /* [N, zp_pitch BYTES] u8|u4 (zero plane) */
    const float* __restrict__ SUM,      /* SUM[g * mpad + m] f32, group-major */
    int M, int N, int K, int mpad, int u4zp,
    int64_t a_pitch, int64_t b_pitch, int64_t y_pitch, int64_t as_pitch,
    int64_t sc_pitch, int64_t zp_pitch)
{
    __shared__ __align__(16) float xs[MT * 512];

    const int w    = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int r    = blockIdx.x * (kThreads / kWave) + w;
    const int rr   = r < N ? r : N - 1;                 /* dead warps load a live row */
    const int live = M < MT ? M : MT;

    const uint8_t* crow = B + (int64_t)rr * b_pitch;
    const int jj = lane >> 3, oo = lane & 7;            /* group in window, lane in group */

    for (int i = threadIdx.x; i < MT * 512; i += kThreads) xs[i] = 0.0f;
    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;
    __syncthreads();

    const int ngrp  = K >> 7;                           /* K % 128 == 0 is the row gate */
    const int nd    = ngrp >> 2;                        /* whole dwords per row */
    const int ntail = ngrp & 3;                         /* whole groups past them */

    /* ---- whole-dword windows: 16 codes per lane, four groups per load ---- */
    for (int wbase = 0; wbase < nd; wbase += W) {
        const int rem = (nd - wbase < W) ? (nd - wbase) : W;
        uint32_t bwin[W];
        if constexpr (W > 1) {
#pragma unroll
            for (int d = 0; d < W; ++d)
                bwin[d] = (d < rem)
                    ? *(const uint32_t*)(crow + (((int64_t)(wbase + d)) << 7) + (lane << 2))
                    : 0u;
        }
        for (int d = 0; d < rem; ++d) {
            const int g0    = (wbase + d) << 2;         /* first group of this dword */
            const int kbase = g0 << 7;                  /* its first code */
            for (int i = threadIdx.x; i < live * 512; i += kThreads) {
                const int m  = i >> 9, kk = i & 511;
                xs[(m << 9) + w2_xs_idx(kk >> 7, kk)] =
                    (float)A[(int64_t)m * a_pitch + kbase + w2_step_pos(kk)];
            }
            __syncthreads();

            const uint32_t b4 = (W > 1) ? bwin[d]
                : *(const uint32_t*)(crow + (((int64_t)wbase + d) << 7) + (lane << 2));
            float sub[MT];
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                float s = 0.0f;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float4 xv = *(const float4*)&xs[(m << 9) + (jj << 7) +
                                                          (u << 5) + (oo << 2)];
                    const float q0 = (float)((b4 >> (8 * u + 0)) & 3u);
                    const float q1 = (float)((b4 >> (8 * u + 2)) & 3u);
                    const float q2 = (float)((b4 >> (8 * u + 4)) & 3u);
                    const float q3 = (float)((b4 >> (8 * u + 6)) & 3u);
                    s = __fmaf_rn(xv.x, q0, s);
                    s = __fmaf_rn(xv.y, q1, s);
                    s = __fmaf_rn(xv.z, q2, s);
                    s = __fmaf_rn(xv.w, q3, s);
                }
                sub[m] = s;                             /* integer-valued, exact */
            }
#pragma unroll
            for (int m = 0; m < MT; ++m) {              /* octet fold: exact at every step */
                sub[m] += __shfl_xor(sub[m], 1);
                sub[m] += __shfl_xor(sub[m], 2);
                sub[m] += __shfl_xor(sub[m], 4);
            }
#pragma unroll
            for (int g2 = 0; g2 < 4; ++g2) {            /* flush ascending */
                const int grp = g0 + g2;
                const float sb = __half2float(SC[(int64_t)rr * sc_pitch + grp]);
                const float z = u4zp
                    ? (float)((ZP[(int64_t)rr * zp_pitch + (grp >> 1)] >> (4 * (grp & 1))) & 0xFu)
                    : (float)ZP[(int64_t)rr * zp_pitch + grp];
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    if (m >= live) break;
                    const float is = __shfl(sub[m], g2 << 3);
                    const float term = z * SUM[(int64_t)grp * mpad + m];
                    acc[m] += ((is - term) * AS[(int64_t)m * as_pitch]) * sb;
                }
            }
            __syncthreads();                            /* xs rewritten next dword */
        }
    }

    /* ---- whole-group tail: byte per lane, the whole wave covers one group ---- */
    for (int t = 0; t < ntail; ++t) {
        const int grp   = (nd << 2) + t;
        const int kbase = grp << 7;
        for (int i = threadIdx.x; i < live * 128; i += kThreads) {
            const int m = i >> 7, kk = i & 127;
            xs[(m << 9) + w2_xs_idx(0, kk)] = (float)A[(int64_t)m * a_pitch + kbase +
                                                       w2_step_pos(kk)];
        }
        __syncthreads();
        const uint8_t bb = crow[(kbase >> 2) + lane];   /* codes kbase + 4*lane + 0..3 */
        float sub[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const float4 xv = *(const float4*)&xs[(m << 9) + ((lane & 3) << 5) +
                                                  ((lane >> 2) << 2)];
            float s = 0.0f;
            s = __fmaf_rn(xv.x, (float)((bb >> 0) & 3u), s);
            s = __fmaf_rn(xv.y, (float)((bb >> 2) & 3u), s);
            s = __fmaf_rn(xv.z, (float)((bb >> 4) & 3u), s);
            s = __fmaf_rn(xv.w, (float)((bb >> 6) & 3u), s);
            sub[m] = s;
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {                  /* full-wave fold: 32 lanes, 1 group */
            sub[m] += __shfl_xor(sub[m], 16);
            sub[m] += __shfl_xor(sub[m], 8);
            sub[m] += __shfl_xor(sub[m], 4);
            sub[m] += __shfl_xor(sub[m], 2);
            sub[m] += __shfl_xor(sub[m], 1);
        }
        const float sb = __half2float(SC[(int64_t)rr * sc_pitch + grp]);
        const float z = u4zp
            ? (float)((ZP[(int64_t)rr * zp_pitch + (grp >> 1)] >> (4 * (grp & 1))) & 0xFu)
            : (float)ZP[(int64_t)rr * zp_pitch + grp];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m >= live) break;
            const float term = z * SUM[(int64_t)grp * mpad + m];
            acc[m] += ((sub[m] - term) * AS[(int64_t)m * as_pitch]) * sb;
        }
        __syncthreads();
    }

    if (lane == 0 && r < N) {
#pragma unroll
        for (int m = 0; m < MT; ++m)
            if (m < live) Y[(int64_t)m * y_pitch + r] = w2_f32_to_bf16(acc[m]);
    }
}

/* ---- the q8 fast arm (hipfire graft, 2026-10-24): v_dot4_i32_iu8 over repacked codes ----
 *
 * The u2 walk above spends its band on VALU: sixteen fmaf per lane-group and four float4
 * LDS reads per row. The microbench that prompted this arm (HIPFIRE-EVAL-2026-10-24, the
 * same warp-per-row geometry with hipfire's q8x inner loop on the XTX) ran 3-5x the
 * libr3d decode GEMVs at every model shape and 5-6x in the M=4 verify band. The inner
 * loop here is that graft: the weight codes repacked to one code a byte (a plane the
 * plugin owns, built once -- see w2_launch's cache), staged i8 activations in LDS, and
 * four __builtin_amdgcn_sudot4 per sixteen codes per lane. The builtin lowers to
 * v_dot4_i32_iu8 with neg_lo:[1,0,0] -- signed i8 activation bytes x unsigned code bytes,
 * i32 accumulate (compile-probed and fuzzed on the card 2026-10-24, together with the
 * nibble-wide v_dot8_i32_iu4, which is the wrong width for this zero-point structure).
 *
 * Bit-exactness survives unchanged: every value up to the scales is an exact integer
 * (the group partial is still |is| <= 128*127*3 < 2^24 -- the dot unit sums four products
 * internally in any order, integers are associative), the octet butterfly folds ints
 * before the float conversion, and the flush from `sub` onward is the oracle's fp chain
 * op for op, byte-identical to the u2 walk's. tools/repro_w2a8.hip adjudicates.
 *
 * The LDS halves (MT x 512 BYTES vs 4x that) and the window loads are dwordx4: sixteen
 * weight bytes and sixteen activation bytes per lane per group, which is also the
 * bytes-in-flight direction the MLP-starvation finding (issue #8 follow-up) names.
 */
__global__ __launch_bounds__(kThreads) void gemm_nt_w2a8_repack_kernel(
    const uint8_t* __restrict__ B, uint8_t* __restrict__ B8,
    int64_t b_pitch, int64_t kb)
{
    const int64_t row = blockIdx.x;
    const int u = blockIdx.y * kThreads + threadIdx.x;      /* sixteen-code units */
    if (u >= (kb >> 2)) return;
    const uint32_t w = *(const uint32_t*)(B + row * b_pitch + ((int64_t)u << 2));
    const uint32_t b0 = w & 0xffu, b1 = (w >> 8) & 0xffu, b2 = (w >> 16) & 0xffu,
                   b3 = w >> 24;
    uint4 o;
    o.x = (b0 & 3u) | ((b0 >> 2 & 3u) << 8) | ((b0 >> 4 & 3u) << 16) | ((b0 >> 6) << 24);
    o.y = (b1 & 3u) | ((b1 >> 2 & 3u) << 8) | ((b1 >> 4 & 3u) << 16) | ((b1 >> 6) << 24);
    o.z = (b2 & 3u) | ((b2 >> 2 & 3u) << 8) | ((b2 >> 4 & 3u) << 16) | ((b2 >> 6) << 24);
    o.w = (b3 & 3u) | ((b3 >> 2 & 3u) << 8) | ((b3 >> 4 & 3u) << 16) | ((b3 >> 6) << 24);
    *(uint4*)(B8 + (row << 2) * b_pitch + ((int64_t)u << 4)) = o;
}

/* crow walks B8 at FOUR times the u2 pitch (the repack preserves row padding); the lane
 * geometry is the u2 walk's own -- lanes 8j..8j+7 cover group g0+j, sixteen ascending
 * codes a lane -- so the octet fold, the ascending flush and the tail's whole-wave fold
 * transfer structurally. xs holds i8 codes at the RAW window position: with byte staging
 * the lane's sixteen codes are sixteen CONSECUTIVE slots, and quant_act_i8's step order
 * still rides the staging index exactly as before. */

/* One dword-window iteration of the q8 walk: stage the 512-code activation window,
 * fold the octet, flush ascending. Shared by BOTH window structures (Q18) so the two
 * arms cannot drift arithmetically -- identical values by construction is a source
 * property, not a bench finding. The b4 operand arrives by value: which walk fetched
 * it (register window or runtime walk) is the ONLY difference between the arms. */
template <int MT>
__device__ __forceinline__ void q8_dword_iter(
    const int8_t* __restrict__ A, int8_t* xs, float* acc, int g0, const uint4& b4,
    int live, int rr, const __half* __restrict__ SC, const uint8_t* __restrict__ ZP,
    const float* __restrict__ SUM, const float* __restrict__ AS, int u4zp,
    int64_t a_pitch, int64_t sc_pitch, int64_t zp_pitch, int mpad, int64_t as_pitch)
{
    const int kbase = g0 << 7;
    for (int i = threadIdx.x; i < live * 512; i += kThreads) {
        const int m  = i >> 9, kk = i & 511;
        xs[(m << 9) + kk] = A[(int64_t)m * a_pitch + kbase + w2_step_pos(kk)];
    }
    __syncthreads();

    const int jj = (threadIdx.x & 31) >> 3, oo = threadIdx.x & 7;
    int32_t isub[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const uint4 av = *(const uint4*)&xs[(m << 9) + (jj << 7) + (oo << 4)];
        /* this ROCm's spelling carries the sign flags in the builtin: (sA, a,
         * sB, b, acc, clamp) -- lowers to v_dot4_i32_iu8 neg_lo:[1,0,0]. */
        int32_t is = __builtin_amdgcn_sudot4(1, (int)av.x, 0, (int)b4.x, 0, 0);
        is = __builtin_amdgcn_sudot4(1, (int)av.y, 0, (int)b4.y, is, 0);
        is = __builtin_amdgcn_sudot4(1, (int)av.z, 0, (int)b4.z, is, 0);
        is = __builtin_amdgcn_sudot4(1, (int)av.w, 0, (int)b4.w, is, 0);
        isub[m] = is;                             /* integer, exact */
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {                /* octet fold: exact */
        isub[m] += __shfl_xor(isub[m], 1);
        isub[m] += __shfl_xor(isub[m], 2);
        isub[m] += __shfl_xor(isub[m], 4);
    }
#pragma unroll
    for (int g2 = 0; g2 < 4; ++g2) {              /* flush ascending */
        const int grp = g0 + g2;
        const float sb = __half2float(SC[(int64_t)rr * sc_pitch + grp]);
        const float z = u4zp
            ? (float)((ZP[(int64_t)rr * zp_pitch + (grp >> 1)] >> (4 * (grp & 1))) & 0xFu)
            : (float)ZP[(int64_t)rr * zp_pitch + grp];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m >= live) break;
            const float is = (float)__shfl(isub[m], g2 << 3);
            const float term = z * SUM[(int64_t)grp * mpad + m];
            acc[m] += ((is - term) * AS[(int64_t)m * as_pitch]) * sb;
        }
    }
    __syncthreads();                              /* xs rewritten next dword */
}

template <int MT, int W>
__global__ __launch_bounds__(kThreads, 2) void gemm_nt_w2a8_gemv_q8_kernel(
    const int8_t* __restrict__ A,
    const uint8_t* __restrict__ B8,     /* repacked codes, pitch = 4 * b_pitch bytes */
    uint16_t* __restrict__ Y,
    const float* __restrict__ AS,
    const __half* __restrict__ SC,
    const uint8_t* __restrict__ ZP,
    const float* __restrict__ SUM,
    int M, int N, int K, int mpad, int u4zp,
    int64_t a_pitch, int64_t b8_pitch, int64_t y_pitch, int64_t as_pitch,
    int64_t sc_pitch, int64_t zp_pitch)
{
    __shared__ __align__(16) int8_t xs[MT * 512];

    const int w    = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int r    = blockIdx.x * (kThreads / kWave) + w;
    const int rr   = r < N ? r : N - 1;
    const int live = M < MT ? M : MT;

    const uint8_t* crow = B8 + (int64_t)rr * b8_pitch;

    for (int i = threadIdx.x; i < MT * 512; i += kThreads) xs[i] = 0;
    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;
    __syncthreads();

    const int ngrp  = K >> 7;
    const int nd    = ngrp >> 2;
    const int ntail = ngrp & 3;

    /* Q18 (spill census, build-head @ 75a4eca): the old shape preloaded the window
     * into `uint4 bwin[W]` but CONSUMED it through the runtime index of a d loop whose
     * trip count is `rem`, so the array could not live in registers and LLVM demoted
     * the whole window to per-thread local memory -- 16x scratch_store_b128 + 1
     * scratch_load site per iteration (the donor's LICM trap, LIBR35D_TRIAGE #1, on our
     * own decode row). Fix: BOTH the preload and the consumption loop carry
     * `#pragma unroll` to the compile-time trip W, so every bwin index is static and a
     * register residence is the only option the compiler has. The d < rem guard is
     * BLOCK-UNIFORM (rem is a function of wbase/nd only), so the barriers inside it
     * keep equal arrival counts across the block. Consumed order, dot order and flush
     * order are untouched: bit-identical by construction, schedule-only.
     *
     * MT GATE (A/B, same card same session, 3 alternating rounds, all five shape
     * classes non-overlapping -- log ~/scratch/r3d/q18_ab.txt): the unrolled body
     * W-fold pays off on the narrow tiles -- M=1 K=8192 +24.7%, M=1 K=16384 +29.5%,
     * M=4 +15.0% -- and LOSES on the wide ones: M=7 (MT=8) -8.1%, M=14 (MT=16)
     * -11.4% (MT=16 arms also grow SGPR spill slots, ssp 178/439 at 118/154 VGPR).
     * The schedule-transfer law is per-geometry here: MT<=4 takes the register
     * window, MT>=8 keeps the verbatim old walk (whose scratch demotion remains the
     * cheaper of the two measured costs at those tiles -- Q18b owns the Wc=4
     * middle-ground probe for the verify band). */
    if constexpr (MT <= 4) {
        for (int wbase = 0; wbase < nd; wbase += W) {
            const int rem = (nd - wbase < W) ? (nd - wbase) : W;
            uint4 bwin[W];
#pragma unroll
            for (int d = 0; d < W; ++d)
                bwin[d] = (d < rem)
                    ? *(const uint4*)(crow + (((int64_t)(wbase + d)) << 9) + (lane << 4))
                    : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
            for (int d = 0; d < W; ++d)
                if (d < rem)
                    q8_dword_iter<MT>(A, xs, acc, (wbase + d) << 2, bwin[d], live, rr,
                                      SC, ZP, SUM, AS, u4zp, a_pitch, sc_pitch,
                                      zp_pitch, mpad, as_pitch);
        }
    } else {
        /* MT >= 8: the verbatim pre-Q18 walk (see the gate note above). */
        for (int wbase = 0; wbase < nd; wbase += W) {
            const int rem = (nd - wbase < W) ? (nd - wbase) : W;
            uint4 bwin[W];
            if constexpr (W > 1) {
#pragma unroll
                for (int d = 0; d < W; ++d)
                    bwin[d] = (d < rem)
                        ? *(const uint4*)(crow + (((int64_t)(wbase + d)) << 9) + (lane << 4))
                        : make_uint4(0u, 0u, 0u, 0u);
            }
            for (int d = 0; d < rem; ++d) {
                const uint4 b4  = (W > 1) ? bwin[d]
                    : *(const uint4*)(crow + (((int64_t)wbase + d) << 9) + (lane << 4));
                q8_dword_iter<MT>(A, xs, acc, (wbase + d) << 2, b4, live, rr, SC, ZP,
                                  SUM, AS, u4zp, a_pitch, sc_pitch, zp_pitch, mpad,
                                  as_pitch);
            }
        }
    }

    for (int t = 0; t < ntail; ++t) {
        const int grp   = (nd << 2) + t;
        const int kbase = grp << 7;
        for (int i = threadIdx.x; i < live * 128; i += kThreads) {
            const int m = i >> 7, kk = i & 127;
            xs[(m << 9) + kk] = A[(int64_t)m * a_pitch + kbase + w2_step_pos(kk)];
        }
        __syncthreads();
        const uint32_t bb = *(const uint32_t*)(crow + kbase + (lane << 2));
        int32_t isub[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m)
            isub[m] = __builtin_amdgcn_sudot4(
                1, (int)*(const uint32_t*)&xs[(m << 9) + (lane << 2)], 0, (int)bb, 0, 0);
#pragma unroll
        for (int m = 0; m < MT; ++m) {                    /* full-wave fold */
            isub[m] += __shfl_xor(isub[m], 16);
            isub[m] += __shfl_xor(isub[m], 8);
            isub[m] += __shfl_xor(isub[m], 4);
            isub[m] += __shfl_xor(isub[m], 2);
            isub[m] += __shfl_xor(isub[m], 1);
        }
        const float sb = __half2float(SC[(int64_t)rr * sc_pitch + grp]);
        const float z = u4zp
            ? (float)((ZP[(int64_t)rr * zp_pitch + (grp >> 1)] >> (4 * (grp & 1))) & 0xFu)
            : (float)ZP[(int64_t)rr * zp_pitch + grp];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m >= live) break;
            const float is = (float)isub[m];
            const float term = z * SUM[(int64_t)grp * mpad + m];
            acc[m] += ((is - term) * AS[(int64_t)m * as_pitch]) * sb;
        }
        __syncthreads();
    }

    if (lane == 0 && r < N) {
#pragma unroll
        for (int m = 0; m < MT; ++m)
            if (m < live) Y[(int64_t)m * y_pitch + r] = w2_f32_to_bf16(acc[m]);
    }
}

}  /* anonymous namespace */

#else  /* non-RDNA3: host-visible no-op */

/* MUST mirror the real
 * kernel's signature exactly: the host pass only ever sees this stub, and the runtime
 * resolves hipLaunchKernelGGL by the MANGLED NAME -- a drifted stub registers a name the
 * device image does not carry, and every instantiation launches as "cannot find symbol /
 * invalid device function" (2026-10-24: the 6th int64 pitch was missing here; the row
 * could never launch on any card and the engine fell back host-side every draft pass). */
namespace {
template <int MT, int W>
__global__ __launch_bounds__(kThreads, 2) void gemm_nt_w2a8_gemv_kernel(
    const int8_t*, const uint8_t*, uint16_t*, const float*, const __half*, const uint8_t*,
    const float*, int, int, int, int, int, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) {}
/* the q8 arm mirrors too (same rule): the repack kernel and the q8 GEMV. */
__global__ void gemm_nt_w2a8_repack_kernel(const uint8_t*, uint8_t*, int64_t, int64_t) {}
template <int MT, int W>
__global__ __launch_bounds__(kThreads, 2) void gemm_nt_w2a8_gemv_q8_kernel(
    const int8_t*, const uint8_t*, uint16_t*, const float*, const __half*, const uint8_t*,
    const float*, int, int, int, int, int, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) {}
}  /* anonymous namespace */

#endif /* gfx11 family */

/* ---- the plan ---------------------------------------------------------------------
 *
 * gemm_nt_q positions: (a, a_scale, b, b_scale, y, a_sum?, b_ref?). Every refusal is
 * RAD_E_UNSUPPORTED so the engine may answer from the host reference (nothing else in
 * libr3d claims a w2a8 band); format gate first, shape gate second, exactly the fp8a8
 * entries' discipline. */

namespace {

struct W2Plan {
    const int8_t*   A;
    const uint8_t*  B;
    const float*    AS;
    const __half*   SC;
    const uint8_t*  ZP;
    const float*    SUM;
    int64_t M, N, K;
    int mpad, u4zp;
    int64_t a_pitch, b_pitch, as_pitch, sc_pitch, zp_pitch;
};

int w2_plan(const RadArgs* a, const RadTensor* A, const RadTensor* AS, const RadTensor* B,
            const RadTensor* BS, const RadTensor* Y, const RadTensor* SUM, W2Plan* P) {
    const char* dtype = rad_args_gets(a, "dtype");
    if (!dtype || strcmp(dtype, "w2a8") != 0) return RAD_E_UNSUPPORTED;

    long long pb1 = 0, pb2 = 0;
    if (rad_args_geti(a, "pb1", &pb1) && pb1 > 0) return RAD_E_UNSUPPORTED;   /* e2m1-only */
    if (rad_args_geti(a, "pb2", &pb2) && pb2 > 0) return RAD_E_UNSUPPORTED;

    long long N = 0, K = 0, gp = 0;
    if (!rad_args_geti(a, "N", &N) || !rad_args_geti(a, "K", &K)) return RAD_E_SHAPE;
    if (N <= 0 || K <= 0) return RAD_E_SHAPE;
    if (!rad_args_geti(a, "group", &gp) || gp != 128) return RAD_E_SHAPE;  /* DF_W2_GROUP */
    if (K % 128) return RAD_E_UNSUPPORTED;             /* the scale grid must close */
    const int64_t M = P->M;                            /* set by the entry: numel(y)/N */
    if (M <= 0 || M > 16) return RAD_E_UNSUPPORTED;     /* the band's own bound, re-checked:
                                                           the drafter declares M to 14, the
                                                           MT=16 tile covers it at 16 */
    if (N % 16) return RAD_E_UNSUPPORTED;              /* libr4d cGemmNtW2A8's N rule */
    if (M > 0x7ffffffLL || N > 0x7ffffffLL || K > 0x7ffffffLL) return RAD_E_SHAPE;

    /* Operand dtypes: gemm_zp's ZP_I8_ROW arm verbatim (ref_gemm.cpp:330 `i8`), plus the
     * f32 ds/dsum the block declares (rad_block_dflash2.h:177-179). */
    if (A->dtype != RAD_I8 || B->dtype != RAD_U2 || AS->dtype != RAD_F32 ||
        Y->dtype != RAD_BF16 || SUM->dtype != RAD_F32)
        return RAD_E_UNSUPPORTED;
    if (rad_tensor_numel(A) < M * K || rad_tensor_numel(B) < N * K ||
        rad_tensor_numel(AS) < M || rad_tensor_numel(Y) < M * N)
        return RAD_E_SHAPE;

    /* quant_act_i8 stores its codes at i8_step_pos positions with a unit last stride;
     * ref refuses anything else (ref_gemm.cpp:415). */
    if (A->rank >= 2 && A->stride[A->rank - 1] != 1) return RAD_E_STRIDE;
    if (B->rank >= 2 && B->stride[B->rank - 1] != 1) return RAD_E_STRIDE;
    if (Y->rank >= 2 && Y->stride[Y->rank - 1] != 1) return RAD_E_STRIDE;

    const int64_t KBYTES = K / 4;                       /* K % 128 == 0: exact bytes */
    const int64_t b_pitch = rad_dtype_bytes(RAD_U2, w2_row_stride(B, K));
    if ((b_pitch % 4) != 0 || (((uintptr_t)B->data) & 3) != 0) return RAD_E_UNSUPPORTED;

    const int64_t mpad = (M + 15) / 16 * 16;            /* the oracle's own mpad */
    const int64_t ngrp = K / 128;
    if ((SUM->rank >= 2 && SUM->stride[SUM->rank - 1] != 1) ||
        rad_tensor_numel(SUM) < ngrp * mpad)            /* ref: numel < G*mpad refuses */
        return RAD_E_SHAPE;

    /* b_scale: the joined "scale,zero" blob our layout hook stored. */
    const RadPlanesHeader* h = rad_planes_header(BS->data);
    if (!h || h->n_planes != 2) return RAD_E_DTYPE;
    if (h->transform[0]) return RAD_E_UNSUPPORTED;      /* the kernel does not rotate */
    const RadPlanesEntry* es = nullptr;
    const RadPlanesEntry* ez = nullptr;
    const void* ps = rad_planes_find(BS->data, "scale", &es);
    const void* pz = rad_planes_find(BS->data, "zero", &ez);
    if (!ps || !pz || !es || !ez) return RAD_E_DTYPE;
    if (es->dtype != RAD_F16) return RAD_E_UNSUPPORTED; /* the checkpoint's affine scale */
    if (ez->dtype != RAD_U8 && ez->dtype != RAD_U4) return RAD_E_UNSUPPORTED;
    if (es->cols != ngrp || es->rows != N || es->block[1] != 128 || es->block[0] != 1)
        return RAD_E_SHAPE;
    if (ez->cols != ngrp || ez->rows != N || ez->block[1] != 128 || ez->block[0] != 1)
        return RAD_E_SHAPE;
    if (es->row_bytes != 2 * ngrp) return RAD_E_STRIDE; /* dense scale rows */
    if (ez->row_bytes != rad_dtype_bytes(ez->dtype, ngrp)) return RAD_E_STRIDE;
    if ((int64_t)h->bytes < ez->offset + ez->rows * ez->row_bytes) return RAD_E_SHAPE;

    /* Chip gate (runtime, not compile-time -- the fp8a8 entries' block, same). */
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

    P->A = (const int8_t*)A->data;   P->B = (const uint8_t*)B->data;
    P->AS = (const float*)AS->data;  P->SUM = (const float*)SUM->data;
    P->SC = (const __half*)((const uint8_t*)BS->data + es->offset);
    P->ZP = (const uint8_t*)BS->data + ez->offset;
    P->N = N; P->K = K; P->mpad = (int)mpad; P->u4zp = (ez->dtype == RAD_U4);
    P->a_pitch  = w2_row_stride(A, K);
    P->b_pitch  = b_pitch;                               /* BYTES (u2: 2 bits a code) */
    P->as_pitch = w2_row_stride(AS, 1);                  /* one scale a row */
    P->sc_pitch = ngrp;                                  /* the joined plane is dense */
    P->zp_pitch = ez->row_bytes;                         /* bytes a zero row */
    return RAD_OK;
}

void w2_launch(const W2Plan& P, void* ydata, int64_t y_pitch, RadStream s) {
    static const bool no_pf = getenv("R3D_NO_GEMV_PF") != nullptr;
    static const int  pf_w  = getenv("R3D_GEMV_PF_W") ? atoi(getenv("R3D_GEMV_PF_W")) : 8;
    static const bool no_q8 = getenv("R3D_NO_W2A8_Q8X") != nullptr;
    hipStream_t stream = (hipStream_t)(uintptr_t)s;
    /* MT rounds UP to cover M: the tile computes min(M, MT) rows and there is no
     * m-loop, so MT < M silently drops rows (serve 2026-10-24: the drafter's block
     * pass collapses the draft rows to 7 -- the old down-round picked MT=4 and
     * lost 3). The band's M<=16 bound makes the 16 tile the ceiling. */
    const int mt = P.M >= 9 ? 16 : P.M >= 5 ? 8 : P.M >= 3 ? 4 : P.M >= 2 ? 2 : 1;
    const dim3 grid((unsigned)((P.N + 7) / 8));

    /* ---- the q8 repack cache (the a16 rule from r3d_gemm_mxfp4.hip: the plugin's OWN
     * buffer, one lazy allocation per draft head, never freed; ANY failure -- alloc,
     * launch, shape ceiling -- falls back to the u2 walk, never a mid-band refusal).
     * Keyed by (codes base, N, K, pitch); four entries cover any pack staging more than
     * one head in a process. The lifetime premise is the band's own: like the joined
     * scale,zero blob and every layout transform before it, checkpoint weight bytes are
     * arena-pinned for the process -- repro regenerates its operands only across runs,
     * never under a live pointer, and the 702-case op gate draws every head fresh per
     * process. A base whose bytes changed under a reused pointer would be caught LOUDLY
     * (bit-exactness dies), not silently. ---- */
    struct Q8Ent { const void* src; uint8_t* dst; int64_t N, K, bp4; };
    static Q8Ent cache[4];
    static int   ncache = 0;
    static bool  q8_dead = no_q8;
    uint8_t* B8 = nullptr;
    if (!q8_dead) {
        for (int i = 0; i < ncache; ++i)
            if (cache[i].src == P.B && cache[i].N == P.N && cache[i].K == P.K &&
                cache[i].bp4 == 4 * P.b_pitch) { B8 = cache[i].dst; break; }
        const int64_t bytes = P.N * 4 * P.b_pitch;          /* u8 codes, pitch kept */
        if (!B8 && ncache < 4 && bytes <= (int64_t)15 << 28) {   /* 1.5 GiB ceiling */
            uint8_t* nb = nullptr;
            if (hipMalloc(&nb, (size_t)bytes) == hipSuccess) {
                /* THE REPACK NEVER TOUCHES THE ENGINE'S STREAM. The drafter's pass
                 * recorder records the issue pattern of the engine's streams on the first
                 * play (kernel launches and stream syncs alike) and asserts the replay
                 * matches; a once-per-process launch inside the recorded stream makes
                 * every later replay short by that issue, and the engine stops the step
                 * loop (observed 2026-10-24: 'draft pass 1 ... issued differently from its
                 * recording'). So the fill runs on the plugin's OWN stream, fully joined
                 * (a host sync on a private stream) before the recorded stream ever gets
                 * its gemv. Weight bytes are arena-pinned long before the first draft
                 * pass, so nothing else needs to be ordered against. */
                static hipStream_t side = nullptr;
                if (!side && hipStreamCreateWithFlags(&side, hipStreamNonBlocking) != hipSuccess)
                    side = nullptr;
                if (side) {
                    const int64_t nu = P.b_pitch >> 2;      /* sixteen-code units a row */
                    dim3 rg((unsigned)P.N, (unsigned)((nu + kThreads - 1) / kThreads));
                    hipLaunchKernelGGL(gemm_nt_w2a8_repack_kernel, rg, dim3(kThreads), 0,
                                       side, P.B, nb, P.b_pitch, P.b_pitch);
                    const hipError_t re = hipGetLastError();
                    if (re == hipSuccess) (void)hipStreamSynchronize(side); /* once, at fill */
                    if (re != hipSuccess || hipGetLastError() != hipSuccess) {
                        (void)hipFree(nb);
                        q8_dead = true;
                        fprintf(stderr, "libr3d: w2a8 repack FAILED, u2 walk takes the band\n");
                    } else {
                        cache[ncache++] = Q8Ent{P.B, nb, P.N, P.K, 4 * P.b_pitch};
                        B8 = nb;
                    }
                } else q8_dead = true;
            } else q8_dead = true;
        }
    }

#define R3D_W2_GEMV(MT_, W_)                                                        \
    hipLaunchKernelGGL((gemm_nt_w2a8_gemv_kernel<MT_, W_>), grid, dim3(kThreads),   \
                       0, stream, P.A, P.B, (uint16_t*)ydata, P.AS, P.SC, P.ZP,    \
                       P.SUM, (int)P.M, (int)P.N, (int)P.K, P.mpad, P.u4zp,        \
                       P.a_pitch, P.b_pitch, y_pitch, P.as_pitch, P.sc_pitch,      \
                       P.zp_pitch)
#define R3D_W2_GEMV_SW(W_)                            \
    switch (mt) {                                     \
        case 1: R3D_W2_GEMV(1, W_); break;            \
        case 2: R3D_W2_GEMV(2, W_); break;            \
        case 4: R3D_W2_GEMV(4, W_); break;            \
        case 8: R3D_W2_GEMV(8, W_); break;            \
        default: R3D_W2_GEMV(16, W_); break;          \
    }
#define R3D_W2_Q8(MT_, W_)                                                       \
    hipLaunchKernelGGL((gemm_nt_w2a8_gemv_q8_kernel<MT_, W_>), grid,             \
                       dim3(kThreads), 0, stream, P.A, B8, (uint16_t*)ydata,     \
                       P.AS, P.SC, P.ZP, P.SUM, (int)P.M, (int)P.N, (int)P.K,    \
                       P.mpad, P.u4zp, P.a_pitch, 4 * P.b_pitch, y_pitch,        \
                       P.as_pitch, P.sc_pitch, P.zp_pitch)
#define R3D_W2_Q8_SW(W_)                            \
    switch (mt) {                                   \
        case 1: R3D_W2_Q8(1, W_); break;            \
        case 2: R3D_W2_Q8(2, W_); break;            \
        case 4: R3D_W2_Q8(4, W_); break;            \
        case 8: R3D_W2_Q8(8, W_); break;            \
        default: R3D_W2_Q8(16, W_); break;          \
    }
    if (B8) {
        if (no_pf)          { R3D_W2_Q8_SW(1) }
        else if (pf_w == 16){ R3D_W2_Q8_SW(16) }
        else                { R3D_W2_Q8_SW(8) }
    }
    else if (no_pf)      { R3D_W2_GEMV_SW(1) }
    else if (pf_w == 16) { R3D_W2_GEMV_SW(16) }
    else                 { R3D_W2_GEMV_SW(8) }
#undef R3D_W2_Q8_SW
#undef R3D_W2_Q8
#undef R3D_W2_GEMV_SW
#undef R3D_W2_GEMV

    const hipError_t le = hipGetLastError();
    if (le != hipSuccess) fprintf(stderr, "libr3d: w2a8 decode gemv launch FAILED (%s)\n",
                                  hipGetErrorName(le));
    else if (getenv("R3D_DEBUG_GEMM")) {
        static bool once = false;
        if (!once) { once = true; fprintf(stderr, "libr3d: w2a8 decode gemv launched (mt=%d, %s)\n",
                                          mt, B8 ? "q8" : "u2"); }
    }
}

}  /* anonymous namespace */

/* ---- the entry --------------------------------------------------------------------
 *
 * The band's whole existence is the DFlash2 drafter, so the kill switch is the
 * drafter family's lever: R3D_NO_DFLASH=1 withdraws the row at the registration gate
 * (tail snippets), and this backstop covers a hand-staged .so whose gate was edited. */

static int r3d_gemm_nt_q_w2a8_dec_impl(const RadArgs*, RadStream);
extern "C" int r3d_rad_gemm_nt_q_w2a8_decode(const RadArgs* a, RadStream s)
    R3D_DUMP_WRAP("gemm_nt_q_w2a8_decode", r3d_gemm_nt_q_w2a8_dec_impl)
static int r3d_gemm_nt_q_w2a8_dec_impl(const RadArgs* a, RadStream s) {
    static const bool no_df = getenv("R3D_NO_DFLASH") != nullptr;
    if (no_df) return RAD_E_UNSUPPORTED;

    const RadTensor* A    = rad_arg_in(a, 0);
    const RadTensor* AS   = rad_arg_in(a, 1);
    const RadTensor* B    = rad_arg_in(a, 2);
    const RadTensor* BS   = rad_arg_in(a, 3);
    const RadTensor* Y    = rad_arg_in(a, 4);
    const RadTensor* ASUM = rad_arg_in(a, 5);
    const RadTensor* BREF = rad_arg_in(a, 6);
    if (!A || !AS || !B || !BS || !Y) return RAD_E_UNSUPPORTED;
    if (!ASUM) return RAD_E_UNSUPPORTED;    /* the zero point needs the group sums (ref:
                                               gemm_zp refuses a ZP_I8_ROW launch without
                                               a_sum, ref_gemm.cpp:334) */
    if (BREF) return RAD_E_UNSUPPORTED;     /* ref accepts b_ref beside an e2m1 weight only */

    /* M is numel(Y)/N -- never the schema's M (the arena-capacity rule). */
    long long Np = 0;
    if (!rad_args_geti(a, "N", &Np) || Np <= 0) return RAD_E_SHAPE;
    const long long Yn = rad_tensor_numel(Y);
    if (Yn <= 0 || Yn % Np) return RAD_E_SHAPE;

    W2Plan P{};
    P.M = Yn / Np;
    int rc = w2_plan(a, A, AS, B, BS, Y, ASUM, &P);
    if (rc) return rc;
    w2_launch(P, Y->data, w2_row_stride(Y, Np), s);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

#endif /* !R3D_W2A8_HOST_GATE */

/* ---- the layout hooks (host-only, no HIP; gate-included) ----------------------------
 *
 * operand 2 (b, the codes plane): stored AS IS -- RAD_E_UNSUPPORTED, the same answer
 *   ref_layout_gemm_q gives (ref_planes.h:192).
 * operand 3 (b_scale, "scale,zero"): the canonical rad_planes.h join, so the bytes are
 *   the ones libref reads (same tag, same header, no plugin-private format). Gates mirror
 *   ref_layout_gemm_q and r4d w2_enc_layout; the DECLINED set is documented in the header.
 */

static int w2_layout_impl(const RadParam* p, int n_p, int operand,
                          const RadEncoding* enc, const int* sel,
                          const RadTensor* planes, int n, RadLayout* out) {
    if (operand != 3 || n < 2 || !out) return RAD_E_UNSUPPORTED;   /* codes and the rest:
                                                                      stored as they are */
    if (!enc || !sel) return RAD_E_DTYPE;
    for (int i = 0; i < n; ++i)
        if (sel[i] < 0 || sel[i] >= enc->n_planes) return RAD_E_DTYPE;
    if (strncmp(enc->scheme, "affine", RAD_ENC_STR) != 0) return RAD_E_DTYPE;
    if (enc->transform[0]) return RAD_E_DTYPE;   /* ref also serves "givens"; this kernel
                                                    does not rotate -- decline at declare
                                                    time, never refuse at issue */
    if (enc->n_planes < 3) return RAD_E_DTYPE;
    /* The selection is the scale and the zero, in that order (RadWeightDecl "scale,zero"). */
    if (strncmp(enc->plane[sel[0]].role, "scale", RAD_ENC_STR) != 0 ||
        strncmp(enc->plane[sel[1]].role, "zero",  RAD_ENC_STR) != 0) return RAD_E_DTYPE;
    const RadEncPlane& c = enc->plane[0];
    const RadEncPlane& s = enc->plane[sel[0]];
    const RadEncPlane& z = enc->plane[sel[1]];
    if (strncmp(c.role, "codes", RAD_ENC_STR) != 0 || c.kind != RAD_PLANE_TILED ||
        c.dtype != RAD_U2 || c.block[0] != 1 || c.block[1] != 1) return RAD_E_DTYPE;
    const long long group = rad_param_geti(p, n_p, "group", 0);
    if (group != 128) return RAD_E_DTYPE;       /* DF_W2_GROUP; no kernel for another */
    if (s.kind != RAD_PLANE_TILED || s.dtype != RAD_F16 ||
        s.block[0] != 1 || s.block[1] != group) return RAD_E_DTYPE;
    if (z.kind != RAD_PLANE_TILED ||
        (z.dtype != RAD_U8 && z.dtype != RAD_U4) ||     /* r4d's accepted set */
        z.block[0] != 1 || z.block[1] != group) return RAD_E_DTYPE;
    const long long N = rad_param_geti(p, n_p, "N", 0);
    const long long K = rad_param_geti(p, n_p, "K", 0);
    if (N <= 0 || K <= 0 || (K % 128)) return RAD_E_SHAPE;
    const long long nsz = K / group;
    /* At declare time the planes carry geometry and a null data (rad_abi.h's RadLayoutFn
     * contract); the round-trip guard is rad_planes_relayout's own data check. */
    for (int i = 0; i < n; ++i)
        if (planes[i].rank < 2 ||
            planes[i].shape[0] != N || planes[i].shape[1] != nsz)
            return RAD_E_SHAPE;                 /* r4d sz_layout's dims(pl[i], N, nsz) */
    return rad_planes_layout(enc, sel, planes, n, out);
}

extern "C" int r3d_layout_gemm_w2a8(const RadParam* p, int n_p, int operand,
                                    const RadEncoding* enc, const int* sel,
                                    const RadTensor* planes, int n, RadLayout* out) {
    const int st = w2_layout_impl(p, n_p, operand, enc, sel, planes, n, out);
    if (getenv("R3D_DEBUG_W2A8")) {
        fprintf(stderr, "W2DBG layout operand=%d n=%d -> %d %s\n", operand, n, st,
                st == RAD_OK && out && out->tag ? out->tag : "");
        for (int i = 0; i < n_p; ++i)
            fprintf(stderr, "  p[%d] %s = %lld %s\n", i, p[i].key, p[i].ival,
                    p[i].sval ? p[i].sval : "");
        if (enc) fprintf(stderr, "  enc scheme=%s n_planes=%d\n", enc->scheme, enc->n_planes);
    }
    return st;
}

extern "C" int r3d_relayout_gemm_w2a8(const RadParam* p, int n_p, int operand,
                                      const RadEncoding* enc, const int* sel,
                                      const RadTensor* planes, int n, void* dst,
                                      int64_t dst_bytes) {
    RadLayout lay{};
    const int st = r3d_layout_gemm_w2a8(p, n_p, operand, enc, sel, planes, n, &lay);
    if (st != RAD_OK) return st;
    return rad_planes_relayout(enc, sel, planes, n, dst, dst_bytes);
}

extern "C" int r3d_unrelayout_gemm_w2a8(const RadParam* p, int n_p, int operand,
                                        const RadEncoding* enc, const int* sel,
                                        const void* src, int64_t src_bytes,
                                        const RadTensor* planes, int n) {
    RadLayout lay{};
    const int st = r3d_layout_gemm_w2a8(p, n_p, operand, enc, sel, planes, n, &lay);
    if (st != RAD_OK) return st;
    return rad_planes_unrelayout(enc, sel, src, src_bytes, planes, n);
}

/* ======================================================================================
 * REGISTRATION SNIPPETS  (apply by hand -- the libr3d repo is NOT edited by this draft)
 * ======================================================================================
 *
 * ---- 1. src/r3d_rows.cpp, beside the other gemm entries (~line 1166) ----------------
 *
 *   extern "C" int r3d_rad_gemm_nt_q_w2a8_decode(const RadArgs*, RadStream);
 *   extern "C" int r3d_layout_gemm_w2a8(const RadParam*, int, int, const RadEncoding*,
 *                                       const int*, const RadTensor*, int, RadLayout*);
 *   extern "C" int r3d_relayout_gemm_w2a8(const RadParam*, int, int, const RadEncoding*,
 *                                         const int*, const RadTensor*, int, void*, int64_t);
 *   extern "C" int r3d_unrelayout_gemm_w2a8(const RadParam*, int, int, const RadEncoding*,
 *                                           const int*, const void*, int64_t,
 *                                           const RadTensor*, int);
 *
 * ---- 2. src/r3d_rows.cpp, beside cMxfp4DecodeRow (~line 976) ------------------------
 *
 * The first three entries are libr4d's cGemmNtW2A8 (r4d_rows.cpp:4495) VERBATIM --
 *   C_DIV("K", 128), C_DIV("N", 16), C_IN("dtype", "w2a8")
 * spelled in this repo's positional form -- and the M/group pair is the band's own, the
 * same law the fp8a8/mxfp4 decode rows carry. r4d's table has NO group predicate (its
 * group is a build flag); ours states it so any other group resolves past the row to the
 * host, which is what r4d's comment argues for ("Any other group gets no kernel rather
 * than a silent 128"). The whole row is the drafter's, so M > 8 -- which r4d serves with
 * its m64 tiles -- DECLINES here: the 27B's only w2a8 band is this M <= 8 draft head.
 *
 *   // gemm_nt_q w2a8 decode band: the DFlash2 draft head (the container declares the
 *   // grid only when the draft weight carries its 2-bit recipe). libr4d's cGemmNtW2A8
 *   // (K DIV 128, N DIV 16, dtype w2a8) plus the decode band M <= 8 and the checkpoint's
 *   // group; the entry re-checks every dtype, the band's bound, and the dword walk.
 *   static const RadConstraint cW2A8DecodeRow[] = {
 *       { "dtype", RAD_C_IN,  0,    "w2a8" },
 *       { "K",     RAD_C_DIV, 128,  nullptr },
 *       { "N",     RAD_C_DIV, 16,   nullptr },
 *       { "M",     RAD_C_LE,  8,    nullptr },
 *       { "group", RAD_C_EQ,  128,  nullptr },
 *   };
 *   enum { R3D_PRIO_W2A8_DECODE = 1100 };   // same law as R3D_PRIO_FP8_DECODE: the
 *                                           // width row owns the band; nothing else in
 *                                           // this plugin claims w2a8 at all
 *
 * ---- 3. src/r3d_rows.cpp, the kRows table after gemm_nt_q_mxfp4_decode (~1406) ------
 *
 *   { "gemm_nt_q_w2a8_decode", "gemm_nt_q", "gemm",
 *     "w2a8 decode-band GEMV (the DFlash2 draft head): warp per weight row, one aligned "
 *     "dword per lane per four scale groups -- sixteen unsigned 2-bit codes per lane -- "
 *     "the int8 activation staged per 512-of-K window through LDS with quant_act_i8's "
 *     "step order de-permuted in the staging index; each group's partial folds through "
 *     "an octet butterfly and flushes as (sub * a_scale) * b_scale in the oracle's "
 *     "order, sub = is - zero * a_sum first -- every value before the scales is an exact "
 *     "f32 integer, so the fold is bit-exact to ref_gemm's gemm_zp, not merely close",
 *     "M <= 8 (the drafter's whole gemm_nt_q band; everything else resolves past the "
 *     "row); K a multiple of 128; N a multiple of 16 (libr4d's rule); group exactly "
 *     "128; M derives from numel(y)/N; a B whose base or row pitch breaks the dword "
 *     "walk resolves past the row, as everything past the band does -- to the host",
 *     "int8 a with an f32 per-row scale (quant_act_i8's step order), u2 b (the codes "
 *     "plane stored as it is) with an f16 scale and a u8/u4 zero per 128 K in the "
 *     "canonical rad.planes join (libref's own layout), f32 a_sum = quant_act_i8's "
 *     "group-major [K/128][Mpad] sums (REQUIRED -- the zero point's correction), bf16 "
 *     "y; b_ref refused; pb1/pb2 refused; an encoding transform declines at declare",
 *     RAD_DOMAIN_DEVICE, R3D_PRIO_W2A8_DECODE,
 *     cW2A8DecodeRow, 5, nullptr, 0, nullptr,
 *     nullptr, nullptr, r3d_rad_gemm_nt_q_w2a8_decode,
 *     nullptr, r3d_layout_gemm_w2a8, r3d_relayout_gemm_w2a8, nullptr,
 *     r3d_unrelayout_gemm_w2a8 },
 *
 * No opd_shape: libref ships the shape fn for the whole conventional op (the logits and
 * mxfp4 rows' precedent). No scratch, no tunables.
 *
 * ---- 4. src/r3d_rows.cpp, the band filter (r3d_band_row_hidden, ~1885) --------------
 *
 * The row exists only for the drafter, so it joins the drafter family the lever already
 * hides (dflash_conv / dflash_select / logit_rerank). One line:
 *
 *   // R3D_NO_DFLASH=1 withdraws the five GDN rows... (unchanged context lines)
 *   -    if (no_df && (!strcmp(k.op, "dflash_conv") || !strcmp(k.op, "dflash_select") ||
 *   -                  !strcmp(k.op, "logit_rerank"))) return true;
 *   +    if (no_df && (!strcmp(k.op, "dflash_conv") || !strcmp(k.op, "dflash_select") ||
 *   +                  !strcmp(k.op, "logit_rerank") ||
 *   +                  !strcmp(k.name, "gemm_nt_q_w2a8_decode")))
 *   +        return true;
 *   (the branch keeps its "drafter family (issue #9)" comment; the draft head row now
 *   rides the same lever)
 *
 * R3D_NO_DFLASH then puts the draft logits band on the host (libref's gemm_zp answers
 * it), which is the serve A/B arm; the entry's own no_df check is a backstop only, for
 * the abort fault the #4 story records.
 *
 * ---- 5. CMakeLists.txt --------------------------------------------------------------
 *
 *   HIP_SOURCES ... src/r3d_gemm_logits.hip  ->  append:  src/r3d_gemm_w2a8.hip
 *
 *   set_source_files_properties(src/r3d_gemm_w2a8.hip
 *     PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")
 *
 * (-ffp-contract=off is also plugin-wide via rad_add_plugin -- pinned here anyway, the
 * house convention for every row that folds f32 chains; the wave32 pin is load-bearing
 * for the octet butterfly's lane arithmetic.)
 *
 * ---- 6. Gate ---------------------------------------------------------------------------------
 *
 * tools/gate.sh needs no change to RUN; it needs a FIXTURE: the stock packs have no
 * w2a8 case (mxfp4_nodraft is nodraft -- no drafter head at all). Record one on a
 * dflash2 2-bit container (or transplant r4d_selftest.cpp:26769's gemm_case line:
 * M in {1,2,4,8}, N = head_rows, K = n_embd, i8 activation, N*K/4 codes, N*(K/128)*4
 * scale bytes), then:
 *
 *   ssh rdna3-dev 'bash ~/projects/libr3d/libr3d/tools/gate.sh op gemm_nt_q gate_w2a8'
 *
 * with R3D_FIXTURE pointing at the new pack, and the A/B arms R3D_NO_DFLASH=1 /
 * R3D_NO_GEMV_PF=1. Until that gate runs on the XTX, this row is UNGATED: do not ship.
 * ===================================================================================== */
