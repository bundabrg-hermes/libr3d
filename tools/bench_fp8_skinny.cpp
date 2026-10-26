// bench_fp8_skinny.cpp -- Q26 A/B: WMMA skinny decode GEMM vs the shipped fp8a8 rows.
//
// Q21 shipped the skinny WMMA pattern for w2a8 dispatch only. This bench asks the
// dtype-cross question for fp8a8 (e4m3 @ e4m3, 128-of-K block scales):
//   * the skinny bundle stages 8 k-slices per barrier vs the m64 tile's 2 per slice;
//   * the register-direct accumulator epilogue skips the m64 flush's Ps LDS round trip;
//   * the deltas vs the w2a8 donor are the E4M3 decode (bf16 LUT, exact embedding --
//     the m64 arm already pays the same decode in prefetch) and the per-group f32
//     flush (no exact-integer class here: this family is tol-graded 3e-02, and the
//     GEMV's dot2 pair-sum order is ALREADY an accepted order within it -- so bitwise
//     ROW==SKINNY is NOT the bar; rel_l2-vs-oracle is, same adjudication as Q9).
//
// Arms in one process, same planes:
//   ROW    = dlopen'd r3d_rad_gemm_nt_q_fp8        (the shipped m64 WMMA tile)
//   GEMV   = dlopen'd r3d_rad_gemm_nt_q_fp8_decode (the shipped decode row, M<=8)
//   NAIVE  = scalar oracle (ref_gemm.h quant_dot order, ascending f32 of exact products)
//   SKINNY = this prototype (bf16 fragments via the e4m3->bf16 LUT, per-group flush)
// Build/run (radiance-dev, card) -- flags mirror the project's correctness law
// (-ffp-contract=off, window 18):
//   hipcc -O2 --offload-arch=gfx1100 -mno-wavefrontsize64 -std=c++17 -ffp-contract=off \
//         -I/opt/radiance/include/radiance -w bench_fp8_skinny.cpp -o /s/f8sk -ldl
//   ROCR_VISIBLE_DEVICES=1 LIBR3D_SO=... /s/f8sk <M> <K> <N> <REPS> <ROT>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <vector>
#include <rocwmma/rocwmma.hpp>
#include "rad_abi.h"
#include "rad_types.h"

namespace roCW = rocwmma;
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP %s: %s\n", #x, hipGetErrorName(e_)); exit(2); } } while (0)

// ---- dtype helpers, verbatim from src/r3d_gemm_fp8a8.hip ---------------------
__device__ __forceinline__ uint16_t f32_to_bf16_f8(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
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
        u = s | 0x7fc00000u;
    } else {
        u = s | ((e - 7u + 127u) << 23) | (m << 20);
    }
    float f;
    memcpy(&f, &u, 4);
    return f;
}
__device__ __forceinline__ float bf16_to_f32_f8(uint16_t h) {
    float f; const uint32_t u = (uint32_t)h << 16; memcpy(&f, &u, 4); return f;
}
constexpr int kLdm = 136;   // bf16-element LDS stride, pad 8 (the w2a8 sk's row-conflict law)

// ---- SKINNY: 16-weight-row WMMA tile per wave, tokens as matrix_b -------------
// Window-26 extension (Q26b residual): M up to 64 via an IN-BLOCK token-slab loop.
// The slab loop sits INSIDE the K-group loop: ws stages once per group, slabs stream
// through xs -- weight traffic stays ONE pass, like the tile's, unlike a grid.y slab
// dimension which would re-walk weights ceil(M/16) times. Accumulators y[slab][8]
// (4 slabs max); LDS footprint unchanged. M<=16 (nslab==1) reduces to the exact
// window-25 arm: same order, same barriers, comparable numbers.
__global__ __launch_bounds__(256) void skinny_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bp,
    uint16_t* __restrict__ Y, const float* __restrict__ AS,
    const uint16_t* __restrict__ BS, int M, int N, int K, long ngrp, int rblk,
    float* __restrict__ dbg = nullptr)
{
    __shared__ uint16_t Lut[256];                            // e4m3 byte -> exact bf16
    __shared__ __align__(16) uint16_t xs[16 * kLdm];         // tokens x slots (bf16)
    __shared__ __align__(16) uint16_t ws[8][16 * kLdm];      // per-wave rows x slots
    if (threadIdx.x < 256) Lut[threadIdx.x] = f32_to_bf16_f8(fp8e4m3_f8((uint8_t)threadIdx.x));
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tile = (blockIdx.x * 8 + w) * 16;              // first weight row of tile
    if (tile >= N) return;                                   // dead waves EXIT
    const int nslab = (M + 15) >> 4;                         // 16-token slabs (<=4)
    const int tok = lane & 15;                               // pinned f32-acc map: col l&15
    __shared__ roCW::bfloat16_t Id[16 * 16];                 // dbg-mode identity scratch
    roCW::fragment<roCW::matrix_a, 16, 16, 16, roCW::bfloat16_t, roCW::row_major> fa;
    roCW::fragment<roCW::matrix_b, 16, 16, 16, roCW::bfloat16_t, roCW::col_major> fb;
    roCW::fragment<roCW::accumulator, 16, 16, 16, float> fc;
    roCW::fragment<roCW::matrix_a, 16, 16, 16, roCW::bfloat16_t, roCW::row_major> fa0;  // dbg identity
    float y[4][8];                                           // per-slab accumulators
    for (int s = 0; s < 4; ++s)
#pragma unroll
        for (int k = 0; k < 8; ++k) y[s][k] = 0.0f;

    for (long g = 0; g < ngrp; ++g) {
        const long k0 = g * 128;
        {   // EVERY wave stages its OWN tile (window-17 law: a tid<128 gate leaves
            // waves 4..7 reading stale LDS -- near-miss values, the dead-row trap).
            // 16 rows x 32 dwords of e4m3 = 512 dwords per wave: 16/lane. Each dword
            // decodes to 4 bf16 (8 bytes) -- a uint2 store, NEVER a uint4: the first
            // build copied the w2a8 bench's uint4 idiom, but there one dword EXPANDS
            // to 16 codes; here it decodes to 4. The 16B store wrote 8B of stack
            // garbage into the neighbouring chunk AND left slots 32..127 of every row
            // staged never -- a quarter-filled tile in a new disguise.
#pragma unroll
            for (int q = 0; q < 16; ++q) {
                const int idx = q * 32 + lane;               // 0..511 within the wave
                const int r = idx >> 5, d = idx & 31, rr = tile + r;
                uint16_t o[4] = {0, 0, 0, 0};
                if (rr < N) {
                    const uint8_t* b = &Bp[(int64_t)rr * K + k0 + d * 4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) o[j] = Lut[b[j]];
                }
                *(uint2*)&ws[w][r * kLdm + d * 4] = *(uint2*)o;
            }
        }
        for (int slab = 0; slab < nslab; ++slab) {           // tokens stream through ws
        const int live_s = M - slab * 16 < 16 ? M - slab * 16 : 16;
        for (int i = threadIdx.x; i < 16 * 128; i += 256) {  // stage this slab (LUT decode)
            const int m = i >> 7, s = i & 127;
            const int mm = slab * 16 + m;
            xs[m * kLdm + s] = (m < live_s) ? Lut[A[(int64_t)mm * K + k0 + s]] : (uint16_t)0;
        }
        __syncthreads();                                     // xs (and ws on slab 0) visible
        roCW::fill_fragment(fc, 0.0f);
#pragma unroll
        for (int kk = 0; kk < 128; kk += 16) {
            roCW::load_matrix_sync(fa, (const roCW::bfloat16_t*)&ws[w][kk], kLdm);
            roCW::load_matrix_sync(fb, (const roCW::bfloat16_t*)&xs[kk], kLdm);
            if (kk == 0) fa0 = fa;
            roCW::mma_sync(fc, fa, fb, fc);
        }
        if (dbg && g == 0 && slab == 0 && blockIdx.x == 0 && w == 0) {              // unit-mode fragment dump
            if (lane == 0) {
                for (int k = 0; k < 8; ++k) dbg[k] = fc.x[k];
                dbg[8] = bf16_to_f32_f8(ws[0][0]); dbg[9] = bf16_to_f32_f8(ws[0][136]);
                dbg[10] = bf16_to_f32_f8(xs[0]);
                long wb = 0, wfirst = -1, xb = 0, xfirst = -1;
                for (int s = 0; s < 128; ++s) {              // FULL ws row 0 check
                    const float v = bf16_to_f32_f8(ws[0][s]);
                    if (!(v == 1.0f)) { ++wb; if (wfirst < 0) wfirst = s; } }
                for (int s = 0; s < 128; ++s) {              // FULL xs row 0 check
                    const float v = bf16_to_f32_f8(xs[s]);
                    if (!(v == 1.0f)) { ++xb; if (xfirst < 0) xfirst = s; } }
                dbg[20] = (float)wb;  dbg[21] = (float)wfirst;   // expect 0, -1
                dbg[22] = (float)xb;  dbg[23] = (float)xfirst;   // expect 0, -1
            }
            if (lane == 16) { for (int k = 0; k < 8; ++k) dbg[24 + k] = fc.x[k]; }
            if (lane == 31) {                                 // wave-0-only scratch write
                for (int j = 0; j < 16; ++j)
                    for (int k = 0; k < 16; ++k) Id[j * 16 + k] = (j == k) ? (roCW::bfloat16_t)1.0f
                                                                           : (roCW::bfloat16_t)0.0f;
            }
            __syncwarp();                                     // wave 0 sees Id (dbg-only)
            if (dbg && g == 0 && slab == 0 && blockIdx.x == 0 && w == 0) {
                // (1) full-wave main-loop check: fc.x[k] == 128 * class(2k + (l>>4)),
                // weight rows cycle classes 1.0/2.0/3.0 by (row % 3) in unit mode.
                int ok = 1;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const float expv = 128.0f * (float)(1 + ((2 * k + (lane >> 4)) % 3));
                    ok &= (fc.x[k] == expv);
                }
                float sum = (float)ok;
                for (int off = 16; off; off >>= 1) sum += __shfl_down(sum, off);
                if (lane == 0) dbg[15] = sum;                // expect 32
                // (2) full-wave row-0 fragment reload: all 32 lanes read the SAME 16x16
                roCW::fragment<roCW::matrix_a, 16, 16, 16, roCW::bfloat16_t, roCW::row_major> fra;
                roCW::fragment<roCW::matrix_a, 16, 16, 16, roCW::bfloat16_t, roCW::row_major> frb;
                const float wv = bf16_to_f32_f8(ws[0][0]);   // unit mode: 1.0
                roCW::load_matrix_sync(fra, (const roCW::bfloat16_t*)&ws[0][0], kLdm);
                roCW::load_matrix_sync(frb, (const roCW::bfloat16_t*)&Id[0], 16);
                float fbad = 0, fbbad = 0;
                for (int k = 0; k < 8; ++k) {
                    // ws rows are UNIFORM per row (unit mode), so any sane A-fragment
                    // map must give value 1+(row%3); the pinned i8 map is row l&15.
                    fbad  += (fra.x[k] != (float)(1 + ((lane & 15) % 3)));
                    // Id loaded as matrix_a: element (row l&15, col k + 8*(l>>4));
                    // 1 exactly on the diagonal.
                    fbbad += (frb.x[k] != (((lane & 15) == k + 8 * (lane >> 4)) ? 1.0f : 0.0f));
                }
                float fsum = fbad, bsum = fbbad;
                for (int off = 16; off; off >>= 1) {
                    fsum += __shfl_down(fsum, off);
                    bsum += __shfl_down(bsum, off);
                }
                if (lane == 0) { dbg[12] = fsum; dbg[13] = bsum; }   // expect 0, 0
                // (3) stride-16 copy of ws rows 0..15 into Id-shaped scratch, reload, mma
            }
            {   // warp-wide identity-B unit mma: fid must come back == fa0
                roCW::fragment<roCW::accumulator, 16, 16, 16, float> fid;
                roCW::fill_fragment(fid, 0.0f);
                roCW::fragment<roCW::matrix_b, 16, 16, 16, roCW::bfloat16_t, roCW::col_major> fbid;
                roCW::load_matrix_sync(fbid, (const roCW::bfloat16_t*)&Id[0], 16);
                roCW::mma_sync(fid, fa0, fbid, fid);
                if (lane == 0) for (int k = 0; k < 8; ++k) dbg[32 + k] = fid.x[k];
            }
        }
        // per-group flush in the m64's oracle order: (sub_g * sa) * sb, ascending g.
        if (tok < live_s) {
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int rr = tile + 2 * k + (lane >> 4);   // pinned f32 map: row 2k+(l>>4)
                if (rr >= N) continue;
                const float sb = bf16_to_f32_f8(BS[(int64_t)(rblk > 1 ? rr / rblk : rr) * ngrp + g]);
                y[slab][k] += (fc.x[k] * AS[(int64_t)(slab * 16 + tok) * ngrp + g]) * sb;
            }
        }
        __syncthreads();                                     // next xs (or ws) stage safe
        }   // slab loop
    }
    for (int slab = 0; slab < nslab; ++slab) {
        const int live_s = M - slab * 16 < 16 ? M - slab * 16 : 16;
        const int mm = slab * 16 + tok;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int rr = tile + 2 * k + (lane >> 4);
            if (tok < live_s && rr < N) Y[(int64_t)mm * N + rr] = f32_to_bf16_f8(y[slab][k]);
        }
    }
}

// ---- NAIVE oracle: one thread per (row, token), ref quant_dot order ------------
__global__ void naive_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bp,
    uint16_t* __restrict__ Y, const float* __restrict__ AS,
    const uint16_t* __restrict__ BS, int M, int N, int K, long ngrp, int rblk)
{
    const int r = blockIdx.x * 256 + threadIdx.x;
    if (r >= N) return;
    const long sr = (rblk > 1) ? (long)(r / rblk) : (long)r;   // the plan's grid fold
    for (int t = 0; t < M; ++t) {
        float acc = 0.0f;
        for (long g = 0; g < ngrp; ++g) {
            float part = 0.0f;                               // ascending f32, exact products
            for (int s = 0; s < 128; ++s) {
                const long k = g * 128 + s;
                part += fp8e4m3_f8(A[(int64_t)t * K + k]) * fp8e4m3_f8(Bp[(int64_t)r * K + k]);
            }
            acc += (part * AS[(int64_t)t * ngrp + g]) * bf16_to_f32_f8(BS[sr * ngrp + g]);
        }
        Y[(int64_t)t * N + r] = f32_to_bf16_f8(acc);
    }
}

// ---- grading -------------------------------------------------------------------
static float bf16_to_f32_h(uint16_t h) { const uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static double rel_l2_h(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = bf16_to_f32_h(a[i]), y = bf16_to_f32_h(b[i]);
        num += (x - y) * (x - y); den += y * y;
    }
    return den > 0 ? std::sqrt(num / den) : (num > 0 ? 1.0 : 0.0);
}
static uint16_t f32_to_bf16_h(float f) {                     // host twin of f32_to_bf16_f8
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static uint8_t f32_to_e4m3_h(float f) {                      // encode magnitudes we fixture
    if (f == 0) return 0;
    const int s = f < 0 ? 0x80 : 0;
    float a = fabsf(f);
    int e = 0; while (a >= 2.0f) { a *= 0.5f; ++e; } while (a < 1.0f && e > 0) { a *= 2.0f; --e; }
    if (e == 0 && a < 1.0f) return s;                        // subnormal fixture: skip
    const int m = (int)lrintf((a - 1.0f) * 8.0f);
    return (uint8_t)(s | ((e + 7) << 3) | m);
}

int main(int argc, char** argv) {
    const int      M    = argc > 1 ? atoi(argv[1]) : 14;
    const long long K   = argc > 2 ? atoll(argv[2]) : 16384;
    const long long N   = argc > 3 ? atoll(argv[3]) : 16384;
    const int      reps = argc > 4 ? atoi(argv[4]) : 30;
    const int      rot  = argc > 5 ? atoi(argv[5]) : 4;
    // F8SK_RBLK=128 builds the checkpoint block grid (BS rows = ceil(N/128));
    // default is the per-row grid (rblk 1).
    const int      rblk = (getenv("F8SK_RBLK") && atoi(getenv("F8SK_RBLK")) == 128) ? 128 : 1;
    if (M < 1 || M > 64 || K <= 0 || N <= 0 || K % 128 || rot < 1 || rot > 4) {
        printf("usage: bench_fp8_skinny M(1..64) K(%%128) N REPS ROT(<=4)\n"); return 2; }

    void* dl = dlopen(getenv("LIBR3D_SO") ? getenv("LIBR3D_SO") : "libr3d.so", RTLD_NOW | RTLD_LOCAL);
    if (!dl) { printf("dlopen libr3d: %s\n", dlerror()); return 1; }
    auto entry_tile = (int (*)(const RadArgs*, RadStream))dlsym(dl, "r3d_rad_gemm_nt_q_fp8");
    auto entry_gemv = (int (*)(const RadArgs*, RadStream))dlsym(dl, "r3d_rad_gemm_nt_q_fp8_decode");
    if (!entry_tile || !entry_gemv) { printf("dlsym entries: %s\n", dlerror()); return 1; }

    const long long ngrp = K / 128;
    const int unit = getenv("F8SK_UNIT") != nullptr;   // deterministic-value unit mode
    uint32_t st = 0x13579bdfu;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };

    // A fixture: up to 64 token rows of e4m3 codes (NaN encodings excluded -- both the
    // kernels and the oracle honor the single NaN code, but a fixture full of NaNs
    // grades nothing).
    std::vector<uint8_t> hA(64 * K);
    if (unit) {
        for (auto& v : hA) v = 0x38;                       // e4m3 +1.0
        if (M <= 14)                                       // padding probe rows: only
            for (int m = 14; m < 16; ++m)                  // dead (never live) at M<=14
                std::fill_n(hA.begin() + (ptrdiff_t)m * K, (size_t)K, (uint8_t)0x00);
    } else {
        for (auto& v : hA) { v = (uint8_t)rnd(); if ((v & 0x7f) == 0x7f) v &= 0x7e; }
    }
    std::vector<uint8_t> hBrot[4];
    // Scales: AS f32 [64, ngrp]; BS bf16 [N, ngrp] per-row grid (rblk 1) or
    // [ceil(N/128), ngrp] checkpoint block grid (rblk 128).
    std::vector<float> hAS(64 * ngrp);
    for (auto& v : hAS) v = unit ? 1.0f : 0.01f + 0.001f * (float)(rnd() % 16);
    std::vector<uint16_t> hBS((size_t)(rblk == 128 ? (N + 127) / 128 : N) * ngrp);
    for (auto& v : hBS) v = unit ? (uint16_t)0x3f80u
                                 : f32_to_bf16_h(0.0030f + 0.000125f * (float)(rnd() % 32));

    void *dAS, *dBS, *dY;
    CK(hipMalloc(&dAS, hAS.size() * 4));
    CK(hipMalloc(&dBS, hBS.size() * 2));
    CK(hipMalloc(&dY, (size_t)64 * N * 2));
    CK(hipMemcpy(dAS, hAS.data(), hAS.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dBS, hBS.data(), hBS.size() * 2, hipMemcpyHostToDevice));

    std::vector<void*> dB(rot), dA(rot);
    // f8 unit weight codes: v, 2v, 4v for v in {0.5, 1.0, 1.5} -- distinct, NaN-free,
    // exact in bf16; weight row rr's value is deterministic per row class.
    static const uint8_t kUnitCodes[3] = { 0x38, 0x40, 0x44 };   // 1.0, 2.0, 3.0
    for (int p = 0; p < rot; ++p) {
        std::vector<uint8_t> hB((size_t)N * K);
        for (auto& v : hB) { v = (uint8_t)rnd(); if ((v & 0x7f) == 0x7f) v &= 0x7e; }
        if (unit) {
            for (size_t r = 0; r < (size_t)N; ++r) {
                const uint8_t c = kUnitCodes[r % 3];           // 1.0/2.0/3.0 by row class
                std::fill_n(hB.begin() + (ptrdiff_t)r * K, (size_t)K, c);
            }
        }
        CK(hipMalloc(&dB[p], hB.size()));
        CK(hipMemcpy(dB[p], hB.data(), hB.size(), hipMemcpyHostToDevice));
        if (p == 0) { CK(hipMalloc(&dA[0], hA.size()));
                      CK(hipMemcpy(dA[0], hA.data(), hA.size(), hipMemcpyHostToDevice)); }
        for (int q = 1; q < rot; ++q) dA[q] = dA[0];
    }

    auto mk = [](void* d, uint32_t dt, int rank, std::vector<int64_t> sh,
                 std::vector<int64_t> stride) {
        RadTensor x{}; x.data = d; x.dtype = dt; x.rank = (uint32_t)rank;
        for (size_t i = 0; i < sh.size(); ++i) { x.shape[i] = sh[i]; x.stride[i] = stride[i]; }
        return x; };
    std::vector<RadTensor> base{
        mk(nullptr, RAD_F8E4M3, 2, {64, K}, {K, 1}),
        mk(dAS, RAD_F32, 2, {64, ngrp}, {ngrp, 1}),
        mk(nullptr, RAD_F8E4M3, 2, {N, K}, {K, 1}),
        mk(dBS, RAD_BF16, 2, {rblk == 128 ? (N + 127) / 128 : N, ngrp}, {ngrp, 1}),
        mk(dY, RAD_BF16, 2, {M, N}, {N, 1}),
    };
    RadTensor* t = nullptr;
    CK(hipHostMalloc((void**)&t, sizeof(RadTensor) * 5));
    RadParam pv[] = { RAD_STR("dtype", "fp8a8"), RAD_INT("N", N),
                      RAD_INT("K", K), RAD_INT("group", 128) };
    auto launch_tile = [&](int p) -> int {
        std::vector<RadTensor> dv = base;
        dv[0].data = dA[p]; dv[2].data = dB[p];
        memcpy(t, dv.data(), sizeof(RadTensor) * 5);
        RadArgs a{}; a.p = pv; a.n_p = 4; a.t = t; a.n_t = 5;
        const int rc = entry_tile(&a, (RadStream)0);
        CK(hipDeviceSynchronize());
        return rc; };
    auto launch_gemv = [&](int p) -> int {
        std::vector<RadTensor> dv = base;
        dv[0].data = dA[p]; dv[2].data = dB[p];
        memcpy(t, dv.data(), sizeof(RadTensor) * 5);
        RadArgs a{}; a.p = pv; a.n_p = 4; a.t = t; a.n_t = 5;
        const int rc = entry_gemv(&a, (RadStream)0);
        CK(hipDeviceSynchronize());
        return rc; };
    auto launch_skinny = [&](int p, float* dbg = nullptr) {
        skinny_kernel<<<(unsigned)((N + 127) / 128), 256>>>(
            (const uint8_t*)dA[p], (const uint8_t*)dB[p], (uint16_t*)dY,
            (const float*)dAS, (const uint16_t*)dBS, M, (int)N, (int)K, ngrp, rblk, dbg);
        CK(hipDeviceSynchronize()); };

    // ---- correctness at plane 0 vs the oracle (tol-graded family: rel_l2 bar) ----
    std::vector<uint16_t> yNv((size_t)M * N), yRo((size_t)M * N), yGv((size_t)M * N), ySk((size_t)M * N);
    const bool trc = getenv("F8SK_TRACE") != nullptr;   // crash-site bisect trace
    #define TRC(s) do { if (trc) { printf("TRC %s\n", s); fflush(stdout); } } while (0)
    TRC("allocs done");
    if (unit) {
        // analytic: every group dot = 128 * wclass(rr); scales 1.0; y = ngrp*128*wclass
        float* ddbg; CK(hipMalloc(&ddbg, 128 * 4));
        CK(hipMemset(dY, 0, (size_t)64 * N * 2));
        launch_skinny(0, ddbg);
        CK(hipMemcpy(ySk.data(), dY, (size_t)M * N * 2, hipMemcpyDeviceToHost));
        float hd[128]; CK(hipMemcpy(hd, ddbg, 128 * 4, hipMemcpyDeviceToHost));
        printf("DBG lane0 fc[0..7]=");
        for (int k = 0; k < 8; ++k) printf("%.0f ", hd[k]);
        printf("| ws00=%.0f ws136=%.0f xs0=%.0f\n", hd[8], hd[9], hd[10]);
        printf("DBG wavemma_ok=%.0f (exp 32) reload_ldm136_bad=%.0f (exp 0) reload_id_bad=%.0f (exp 0)\n",
               hd[15], hd[12], hd[13]);
        printf("DBG lane16 fc=");
        for (int k = 0; k < 8; ++k) printf("%.0f ", hd[24 + k]);
        printf("| idmma fid[0..7]=");
        for (int k = 0; k < 8; ++k) printf("%.0f ", hd[32 + k]);
        printf("\n");
        {   // row-check results: ws row0 bad/first-bad, xs row0 bad/first-bad
            printf("ROWCHECK ws row0 bad=%.0f first=%.0f | xs row0 bad=%.0f first=%.0f\n",
                   hd[20], hd[21], hd[22], hd[23]);
        }
        long ok = 0;
        for (size_t i = 0; i < ySk.size(); ++i) {
            const size_t r = i % (size_t)N;
            const float expect = (float)(ngrp * 128) * (float)(1 + (r % 3));
            ok += (bf16_to_f32_h(ySk[i]) == expect);
        }
        printf("UNIT SKINNY analytic %ld/%zu\n", ok, ySk.size());
        return 0;
    }
    naive_kernel<<<(unsigned)((N + 255) / 256), 256>>>(
        (const uint8_t*)dA[0], (const uint8_t*)dB[0], (uint16_t*)dY,
        (const float*)dAS, (const uint16_t*)dBS, M, (int)N, (int)K, ngrp, rblk);
    TRC("naive launched");
    CK(hipDeviceSynchronize());
    TRC("naive synced");
    CK(hipMemcpy(yNv.data(), dY, (size_t)M * N * 2, hipMemcpyDeviceToHost));
    TRC("naive readback");
    long bad = 0; for (auto v : yNv) if (std::isnan(bf16_to_f32_h(v))) ++bad;
    printf("ORACLE NaN outputs %ld/%zu (fixture must hold NaN-free: 0)\n", bad, yNv.size());
    fflush(stdout);

    if (const int rc = launch_tile(0)) { printf("TILE ENTRY REFUSED rc=%d\n", rc); return 3; }
    TRC("tile launched");
    CK(hipMemcpy(yRo.data(), dY, (size_t)M * N * 2, hipMemcpyDeviceToHost));
    TRC("tile readback");
    int gemv_rc = -1;
    if (M <= 8) {
        CK(hipMemset(dY, 0, (size_t)64 * N * 2));
        gemv_rc = launch_gemv(0);
        if (gemv_rc == 0) CK(hipMemcpy(yGv.data(), dY, (size_t)M * N * 2, hipMemcpyDeviceToHost));
    }
    CK(hipMemset(dY, 0, (size_t)64 * N * 2));
    launch_skinny(0);
    CK(hipMemcpy(ySk.data(), dY, (size_t)M * N * 2, hipMemcpyDeviceToHost));

    long sk_eq = 0, ro_eq = 0;
    for (size_t i = 0; i < yNv.size(); ++i) { sk_eq += (ySk[i] == yNv[i]); ro_eq += (yRo[i] == yNv[i]); }
    printf("CHECK rblk=%d M=%d K=%lld N=%lld: ROW==oracle %ld/%zu rel_l2=%.2e | SKINNY==oracle %ld/%zu rel_l2=%.2e\n",
           rblk, M, K, N, ro_eq, yNv.size(), rel_l2_h(yRo, yNv), sk_eq, yNv.size(), rel_l2_h(ySk, yNv));
    if (getenv("F8SK_CROSS")) {
        /* Q26c dispatch-vs-kernel bitwise probe. An earlier draft of this probe
         * compared token-row SLICES of the M=32 output to each other expecting
         * bitwise equality -- FALSE PREMISE, caught by its own result: fixture A
         * rows are independent randoms, rows 16..31 are NOT rows 0..15. The valid
         * equality in a tol-graded family is two INDEPENDENT implementations:
         * the dispatch path (library, now riding the skinny band) vs the bench's
         * own kernel, same fixtures, same order. */
        long bit_eq = 0;
        for (size_t i = 0; i < yNv.size(); ++i) bit_eq += (yRo[i] == ySk[i]);
        printf("CROSS M=%d dispatch(ROW)==bench-kernel(SKINNY) BITWISE %ld/%zu\n",
               M, bit_eq, yNv.size());
    }
    {   // mismatch anatomy: NaN counts + which weight rows / token cols are wrong
        long nan = 0;
        // histogram must cover 64 token-rows + ceil(N/256) weight bins: at
        // N=24576 the fixed 128 overran the heap (glibc 'corrupted size' abort --
        // a TOOL bug that masked the library's verdict at battery shapes).
        std::vector<long> badrow(64 + (size_t)((N + 255) / 256), 0);
        for (size_t i = 0; i < yNv.size(); ++i) {
            if (std::isnan(bf16_to_f32_h(ySk[i]))) ++nan;
            if (ySk[i] == yNv[i]) continue;
            const size_t t = i / (size_t)N, r = i % (size_t)N;
            if (t < 64) ++badrow[t];                // token-row histogram
            ++badrow[64 + (r >> 8)];                // weight row/256 histogram
        }
        printf("ANAT SKINNY nan=%ld tokhist=", nan);
        for (int t = 0; t < 64; ++t) printf("%d ", (int)badrow[t]);
        printf("| row/256hist=");
        for (size_t b = 0; b + 64 < badrow.size(); ++b) printf("%d ", (int)badrow[64 + b]);
        printf("\n");
        long shown = 0;
        for (size_t i = 0; i < yNv.size() && shown < 8; ++i)
            if (ySk[i] != yNv[i]) {
                printf("  [t=%zu r=%zu] SK=%04x RO=%04x NV=%04x\n", i / (size_t)N, i % (size_t)N,
                       ySk[i], yRo[i], yNv[i]); ++shown; }
    }
    if (gemv_rc == 0) {
        long gv_eq = 0; for (size_t i = 0; i < yNv.size(); ++i) gv_eq += (yGv[i] == yNv[i]);
        printf("      GEMV==oracle %ld/%zu rel_l2=%.2e\n", gv_eq, yNv.size(), rel_l2_h(yGv, yNv));
    } else if (M <= 8) printf("      GEMV ENTRY REFUSED rc=%d\n", gemv_rc);

    // ---- alternating A/B timing (card protocol), same planes, same reps ----
    for (int p = 0; p < rot; ++p) { launch_tile(p); launch_skinny(p); if (M <= 8) launch_gemv(p); }
    auto timeit = [&](int which, double* med, double* p2, double* p8) {
        std::vector<double> ts;
        for (int r = 0; r < reps; ++r) {
            timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            if (which == 0) launch_tile(r % rot);
            else if (which == 1) launch_skinny(r % rot);
            else launch_gemv(r % rot);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            ts.push_back((t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec));
        }
        std::sort(ts.begin(), ts.end());
        *med = ts[ts.size() / 2]; *p2 = ts[ts.size() / 5]; *p8 = ts[4 * ts.size() / 5]; };
    double mt, msk, mgv, p2t, p8t, p2s, p8s, p2g, p8g;
    timeit(0, &mt, &p2t, &p8t);
    timeit(1, &msk, &p2s, &p8s);
    const double gb = (double)N * K / 1e9;
    printf("A TILE   M=%d rot=%d | t_med=%.3f ms p2=%.3f p8=%.3f | %.1f GB/s e4m3\n",
           M, rot, mt * 1e3, p2t * 1e3, p8t * 1e3, gb / mt);
    printf("B SKINNY M=%d rot=%d | t_med=%.3f ms p2=%.3f p8=%.3f | %.1f GB/s e4m3\n",
           M, rot, msk * 1e3, p2s * 1e3, p8s * 1e3, gb / msk);
    printf("DELTA SKINNY/TILE M=%d | (A/B-1)*100 = %+.1f%%\n", M, (mt / msk - 1.0) * 100.0);
    if (M <= 8) {
        timeit(2, &mgv, &p2g, &p8g);
        printf("C GEMV   M=%d rot=%d | t_med=%.3f ms p2=%.3f p8=%.3f | %.1f GB/s e4m3\n",
               M, rot, mgv * 1e3, p2g * 1e3, p8g * 1e3, gb / mgv);
        printf("DELTA TILE/GEMV M=%d | (C/A-1)*100 = %+.1f%%\n", M, (mgv / mt - 1.0) * 100.0);
        printf("DELTA SKINNY/GEMV M=%d | (C/B-1)*100 = %+.1f%%\n", M, (mgv / msk - 1.0) * 100.0);
    }
    (void)f32_to_e4m3_h;
    return 0;
}
