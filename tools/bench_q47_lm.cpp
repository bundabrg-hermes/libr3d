// bench_q47_lm.cpp -- Q47 card pass: Arm-A register-tile GEMV window (bench-first).
//
// Desk-booked shape (Q47_DESK_REPORT §6): the shipped Q44 lane-map walk
// gemm_nt_mxfp4_gemv_q8x_lm_kernel<MT> rides a WIN=2 cur/nxt staging window on the
// 136B-block weight stream (2x uint4 codes + 1 scale byte), Q7-style ping-pong on
// NAMED locals (Q18 static-index law: no arrays, no dynamic index), ROWS=1, grid
// untouched (one wave32 per weight row). WIN=1 = the verbatim shipped walk = the
// withdrawal arm. Schedule-only by construction: no dot leg, scale CONVERT, flush
// leg, or tail move -> bit-identity expected (Q54 WIN precedent, Q44 EQUIV 504/504).
//
// Arms in ONE process, same operands:
//   lib  = dlopen'd r3d_rad_gemm_nt_q_mxfp4_decode_q8x (the SHIPPED walk: library's
//          own repack plane + act prologue + lm launch; default levers nt=1 dpp=1
//          lmap=1 -> the lm arm at mt>=4 is what serves).
//   base = this TU's verbatim copy of the lm walk (q47_lm_kernel<MT,1> == src@1342
//          instruction order), fed by this TU's verbatim copies of
//          m4_repack_q8x_kernel and m4_act_i8_kernel on the same operand planes.
//          FIDELITY GATE: bitwise == lib per leg (Q21b precedent: -ffp-contract=off
//          both sides makes bench-vs-library bitwise equal).
//   win  = the candidate q47_lm_kernel<MT,2> (the window). EQUIV gate: bitwise == base.
// Timing: per-arm reps cycling rot distinct planes; the q8x plane bytes
// (N*(K/128)*136, the stream the walk actually reads) are the rate basis.
// rot=4 = COLD (4 planes past the 96 MB INFINITY CACHE); rot=1 with a plane under
// 96 MB = cache-fit. The regime is printed on every leg line.
//
// Row-write gate (the latent M 5..7 gap): the mt ladder is the library's own
// (mt = M>=8?8:M>=4?4:M>=2?2:1) and live=min(M,MT), so M 5..7 @ mt=4 NEVER writes
// rows 4..M-1 -- every arm mirrors the gap, none widens it. Y is sentinel-poisoned
// before every correctness launch; per-row write counts are asserted per leg.
//
// Build/run (radiance-dev ON the card; project correctness flags, window-18 law):
//   hipcc -O2 --offload-arch=gfx1100 -mno-wavefrontsize64 -std=c++17 -ffp-contract=off \
//         -I/opt/radiance/include/radiance -w bench_q47_lm.cpp -o bw2q47 -ldl
//   ROCR_VISIBLE_DEVICES=1 LIBR3D_SO=... Q47_ROUND=<n> [Q47_REV=1] ./bw2q47 <M> <K> <N> <REPS> <ROT>
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <vector>
#include "rad_abi.h"
#include "rad_types.h"

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP %s: %s\n", #x, hipGetErrorName(e_)); exit(2); } } while (0)

// ---- helpers, verbatim from src/r3d_gemm_mxfp4.hip + src/r3d_mxfp4_q8x.h --------
__device__ __forceinline__ uint16_t f32_to_bf16_m4(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
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
__device__ __forceinline__ float r3d_e8m0_sc(uint8_t x) {                 // q8x.h verbatim
    uint32_t u = 0x7fc00000u;
    if (x != 0xffu) u = (x == 0x00u) ? 0x00400000u : ((uint32_t)x << 23);
    float f;
    unsigned char* d = (unsigned char*)&f;
    const unsigned char* s = (const unsigned char*)&u;
    for (int i = 0; i < 4; ++i) d[i] = s[i];
    return f;
}
__device__ __forceinline__ uint32_t q8x_perm_x16(uint32_t v) {
    return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xFEDCBA98u, false, false);
}
#define R3D_E2M1_X2  { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 }

// ---- m4_repack_q8x_kernel, verbatim from src/r3d_gemm_mxfp4.hip:1114 ------------
__global__ __launch_bounds__(256) void m4_repack_q8x_kernel(
    const uint8_t* __restrict__ B, const uint8_t* __restrict__ BS, uint8_t* __restrict__ S,
    int64_t b_pitch, int64_t bs_pitch, int N, int K)
{
    constexpr int8_t lut[16] = R3D_E2M1_X2;
    const int K128 = K >> 7;                       /* entry gate: K % 256 == 0 */
    const int64_t rpitch = (int64_t)K128 * 136;
    const int64_t ncodes = (int64_t)N * (K >> 1);  /* one thread per nibble BYTE */
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < ncodes + (int64_t)N * K128; i += (int64_t)gridDim.x * blockDim.x) {
        if (i < ncodes) {
            const int64_t n = i / (K >> 1);
            const int b = (int)(i - n * (K >> 1));
            const uint8_t byte = B[n * b_pitch + b];
            const uint16_t pair = (uint16_t)(uint8_t)lut[byte & 15u]
                                | ((uint16_t)(uint8_t)lut[byte >> 4] << 8);
            *(uint16_t*)(S + n * rpitch + (b >> 6) * 136 + 8 + (b & 63) * 2) = pair;
        } else {                                   /* group header: 4 scales, 4 pad */
            const int64_t j = i - ncodes;
            const int64_t n = j / K128;
            const int g = (int)(j - n * K128);
            uint8_t* h = S + n * rpitch + g * 136;
            const uint8_t* sb = BS + n * bs_pitch + g * 4;
            h[0] = sb[0]; h[1] = sb[1]; h[2] = sb[2]; h[3] = sb[3];
            h[4] = 0; h[5] = 0; h[6] = 0; h[7] = 0;
        }
    }
}

// ---- m4_act_i8_kernel, verbatim from src/r3d_gemm_mxfp4.hip:1150 ----------------
__global__ __launch_bounds__(256) void m4_act_i8_kernel(
    const uint8_t* __restrict__ A, int8_t* __restrict__ X8, float* __restrict__ SG,
    int M, int K, int64_t a_pitch)
{
    const int cols = K >> 5;
    const int gwarp = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (gwarp >= M * cols) return;
    const int m = gwarp / cols, g = gwarp - m * cols;
    const int lane = threadIdx.x & 31;
    const float e = fp8e4m3_gemv(A[(int64_t)m * a_pitch + (g << 5) + lane]);
    float mx = fabsf(e);
#pragma unroll
    for (int off = 16; off; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off));
    const float s = mx ? mx / 127.0f : 1.0f;
    if (lane == 0) SG[(int64_t)m * cols + g] = s;
    X8[(int64_t)m * K + (g << 5) + lane] = (int8_t)rintf(e / s);
}

// ---- lm walk <MT, WIN>: the shipped Q44 lane-map kernel + the Q47 window ---------
// WIN=1 is the src/r3d_gemm_mxfp4.hip:1342 walk verbatim (fidelity control).
// WIN=2 issues iteration it+1's (wa, wbv, scale byte) into named cur/nxt locals
// BEFORE iteration it's m-loop consumes its operands: the load->fold dependency
// Q50 attributed as the walk's residual (chain ~ MT) gets a full iteration of
// lead time. Q7 parity ping-pong on NAMED locals: no array, no dynamic index.
// Nothing else moves: g, the scale CONVERT, the sudot4 chain, the flush product,
// the acc order, the peel, and the tail butterfly stay at shipped positions.
template <int MT, int WIN>
__global__ __launch_bounds__(32, 16) void q47_lm_kernel(
    const uint8_t* __restrict__ S, const int8_t* __restrict__ X, const float* __restrict__ SG,
    const float* __restrict__ AS, uint16_t* __restrict__ Y,
    int M, int N, int K, int64_t y_pitch, int64_t as_pitch, int nt, int dpp)
{
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    const int nfull = K >> 10;                 /* full 1024-code iterations */
    const int ngrp  = K >> 5;                  /* total 32-code groups on the row */
    const int live  = M < MT ? M : MT;
    const uint8_t* crow = S + (int64_t)r * ((int64_t)(K >> 7) * 136);

    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

    const int wb = t >> 2;                     /* owned 136B block within the 8-span */
    const int ws = (t & 3) * 32;               /* owned sub-block byte offset */

    // One iteration's weight burst: 2x16B codes + the lane's scale byte. The macro
    // evaluates an expression-time iteration index only -- never an array index.
#define Q47_LOADW(wa_, wbv_, sb_, it_) do {                                        \
        const uint8_t* blk_ = crow + ((int64_t)(it_) << 3) * 136;                   \
        const uint8_t* wp_  = blk_ + wb * 136 + 8 + ws;                             \
        if (nt) {                              /* 4x b64 NT: same hint as shipped */\
            const uint64_t r0 = __builtin_nontemporal_load((const uint64_t*)wp_);   \
            const uint64_t r1 = __builtin_nontemporal_load((const uint64_t*)(wp_ + 8)); \
            const uint64_t r2 = __builtin_nontemporal_load((const uint64_t*)(wp_ + 16));\
            const uint64_t r3 = __builtin_nontemporal_load((const uint64_t*)(wp_ + 24));\
            (wa_)  = make_uint4((uint32_t)r0, (uint32_t)(r0 >> 32),                 \
                                (uint32_t)r1, (uint32_t)(r1 >> 32));                \
            (wbv_) = make_uint4((uint32_t)r2, (uint32_t)(r2 >> 32),                 \
                                (uint32_t)r3, (uint32_t)(r3 >> 32));                \
        } else {                                                                   \
            (wa_)  = *(const uint4*)wp_;                                           \
            (wbv_) = *(const uint4*)(wp_ + 16);                                     \
        }                                                                          \
        (sb_) = blk_[wb * 136 + (t & 3)];                                          \
    } while (0)

    uint4 wa_c = make_uint4(0,0,0,0), wbv_c = make_uint4(0,0,0,0);
    uint4 wa_n = make_uint4(0,0,0,0), wbv_n = make_uint4(0,0,0,0);
    uint8_t sb_c = 0, sb_n = 0;
    if constexpr (WIN > 1) { if (nfull > 0) Q47_LOADW(wa_c, wbv_c, sb_c, 0); }

    for (int it = 0; it < nfull; ++it) {
        uint4 wa, wbv;
        uint8_t sb;
        if constexpr (WIN > 1) {
            if (it + 1 < nfull) Q47_LOADW(wa_n, wbv_n, sb_n, it + 1); /* in flight */
            wa = wa_c; wbv = wbv_c; sb = sb_c;                        /* consume CURRENT */
            wa_c = wa_n; wbv_c = wbv_n; sb_c = sb_n;                  /* static reg copy */
        } else {
            Q47_LOADW(wa, wbv, sb, it);                               /* verbatim walk */
        }
        const int g = (it << 5) + t;           /* the LANE owns the 32-group */
        const float sc = r3d_e8m0_sc(sb);      /* CONVERT at the shipped position */
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m >= live) break;              /* block-uniform: every lane skips */
            const int8_t* xp = X + (int64_t)m * K + ((int64_t)it << 10) + t * 32;
            const uint4 xa = *(const uint4*)xp;
            const uint4 xb = *(const uint4*)(xp + 16);
            int32_t p = __builtin_amdgcn_sudot4(1, (int)xa.x, 1, (int)wa.x, 0, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xa.y, 1, (int)wa.y, p, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xa.z, 1, (int)wa.z, p, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xa.w, 1, (int)wa.w, p, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xb.x, 1, (int)wbv.x, p, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xb.y, 1, (int)wbv.y, p, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xb.z, 1, (int)wbv.z, p, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xb.w, 1, (int)wbv.w, p, 0);
            float c = (float)p * SG[(int64_t)m * (K >> 5) + g]
                            * AS[(int64_t)m * as_pitch + (g >> 2)] * sc * 0.5f;
            acc[m] += c;                       /* per-lane containment: NO fold */
        }
    }
    /* ragged-K peel (K%1024 tail) -- verbatim, OUTSIDE the window: nfull*32 groups
     * close every 1024; at most one extra group per lane, schedule-only law says
     * do not touch it. */
    for (int g = (nfull << 5) + t; g < ngrp; g += 32) {
        const uint8_t* blk = crow + (int64_t)(g >> 2) * 136;
        const uint8_t* wp = blk + 8 + ((g & 3) << 5);
        uint4 wa, wbv;
        if (nt) {
            const uint64_t r0 = __builtin_nontemporal_load((const uint64_t*)wp);
            const uint64_t r1 = __builtin_nontemporal_load((const uint64_t*)(wp + 8));
            const uint64_t r2 = __builtin_nontemporal_load((const uint64_t*)(wp + 16));
            const uint64_t r3 = __builtin_nontemporal_load((const uint64_t*)(wp + 24));
            wa = make_uint4((uint32_t)r0, (uint32_t)(r0 >> 32), (uint32_t)r1, (uint32_t)(r1 >> 32));
            wbv = make_uint4((uint32_t)r2, (uint32_t)(r2 >> 32), (uint32_t)r3, (uint32_t)(r3 >> 32));
        } else {
            wa = *(const uint4*)wp;
            wbv = *(const uint4*)(wp + 16);
        }
        const float sc = r3d_e8m0_sc(blk[g & 3]);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m >= live) break;
            const int8_t* xp = X + (int64_t)m * K + ((int64_t)g << 5);
            const uint4 xa = *(const uint4*)xp;
            const uint4 xb = *(const uint4*)(xp + 16);
            int32_t p = __builtin_amdgcn_sudot4(1, (int)xa.x, 1, (int)wa.x, 0, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xa.y, 1, (int)wa.y, p, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xa.z, 1, (int)wa.z, p, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xa.w, 1, (int)wa.w, p, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xb.x, 1, (int)wbv.x, p, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xb.y, 1, (int)wbv.y, p, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xb.z, 1, (int)wbv.z, p, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xb.w, 1, (int)wbv.w, p, 0);
            float c = (float)p * SG[(int64_t)m * (K >> 5) + g]
                            * AS[(int64_t)m * as_pitch + (g >> 2)] * sc * 0.5f;
            acc[m] += c;
        }
    }
    /* single tail reduce: full-wave butterfly at row end -- verbatim */
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        float v = acc[m];
        if (dpp) {
            v += __uint_as_float(__builtin_amdgcn_update_dpp(
                __float_as_uint(v), __float_as_uint(v), 0x161, 0xf, 0xf, 0));
            v += __uint_as_float(__builtin_amdgcn_update_dpp(
                __float_as_uint(v), __float_as_uint(v), 0x162, 0xf, 0xf, 0));
            v += __uint_as_float(__builtin_amdgcn_update_dpp(
                __float_as_uint(v), __float_as_uint(v), 0x164, 0xf, 0xf, 0));
            v += __uint_as_float(__builtin_amdgcn_update_dpp(
                __float_as_uint(v), __float_as_uint(v), 0x168, 0xf, 0xf, 0));
            v += dpp >= 2 ? __uint_as_float(q8x_perm_x16(__float_as_uint(v)))
                          : __shfl_xor(v, 16);
        } else {
            v += __shfl_xor(v, 1);
            v += __shfl_xor(v, 2);
            v += __shfl_xor(v, 4);
            v += __shfl_xor(v, 8);
            v += __shfl_xor(v, 16);
        }
        if (t == 0 && m < live) Y[(int64_t)m * y_pitch + r] = f32_to_bf16_m4(v);
    }
#undef Q47_LOADW
}

// ---- host side -------------------------------------------------------------------
static uint64_t fnv1a(const std::vector<uint16_t>& v) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t* b = (const uint8_t*)v.data();
    for (size_t i = 0; i < v.size() * 2; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
static size_t nz16(const std::vector<uint16_t>& v) {
    size_t n = 0; for (auto h : v) n += ((h & 0x7fffu) != 0); return n;
}
static const uint16_t kSentinel = 0xA5A5u;

int main(int argc, char** argv) {
    const int      M    = argc > 1 ? atoi(argv[1]) : 4;
    const long long K   = argc > 2 ? atoll(argv[2]) : 4096;
    const long long N   = argc > 3 ? atoll(argv[3]) : 12288;
    const int      reps = argc > 4 ? atoi(argv[4]) : 24;
    const int      rot  = argc > 5 ? atoi(argv[5]) : 4;
    const int      RND  = getenv("Q47_ROUND") ? atoi(getenv("Q47_ROUND")) : 0;
    const bool     REV  = getenv("Q47_REV") && atoi(getenv("Q47_REV")) != 0;
    if (M < 1 || M > 8 || K <= 0 || N <= 0 || (K % 256) || rot < 1 || rot > 4) {
        printf("usage: bench_q47_lm M(1..8) K(%%256) N REPS ROT(<=4)\n"); return 2; }

    void* dl = dlopen(getenv("LIBR3D_SO") ? getenv("LIBR3D_SO") : "libr3d.so",
                      RTLD_NOW | RTLD_LOCAL);
    if (!dl) { printf("dlopen libr3d: %s\n", dlerror()); return 1; }
    auto entry = (int (*)(const RadArgs*, RadStream))
                     dlsym(dl, "r3d_rad_gemm_nt_q_mxfp4_decode_q8x");
    if (!entry) { printf("dlsym entry: %s\n", dlerror()); return 1; }

    const int mt = M >= 8 ? 8 : M >= 4 ? 4 : M >= 2 ? 2 : 1;   // the library ladder verbatim
    const int live = M < mt ? M : mt;
    const long long as_cols = (K + 127) / 128, bs_cols = (K + 31) / 32;
    const long long rpitch = (K >> 7) * 136;                    // q8x plane row pitch
    const double plane_bytes = (double)N * (double)rpitch;      // the streamed plane
    const char* regime = (plane_bytes * rot > 96e6) ? "cold" : "cache-fit";

    uint32_t st = 0x9e3779b9u ^ (uint32_t)(M * 131 + K * 7 + N);
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };

    std::vector<uint8_t> hA((size_t)M * K);
    for (auto& v : hA) { v = (uint8_t)rnd(); if ((v & 0x7f) == 0x7f) v &= 0x7e; }
    std::vector<float> hAS((size_t)M * as_cols);
    for (auto& v : hAS) v = 0.01f + 0.001f * (float)(rnd() % 16);
    std::vector<uint8_t> hBS((size_t)N * bs_cols);
    for (auto& v : hBS) v = (uint8_t)(115 + rnd() % 25);        // e8m0, never 0x00/0xFF

    void *dA, *dAS, *dBS, *dY, *dX8, *dSG;
    CK(hipMalloc(&dA,  hA.size()));
    CK(hipMalloc(&dAS, hAS.size() * 4));
    CK(hipMalloc(&dBS, hBS.size()));
    CK(hipMalloc(&dY,  (size_t)8 * N * 2));                     // arena-capacity rows
    CK(hipMalloc(&dX8, (size_t)8 * K));                         // bench-side act scratch
    CK(hipMalloc(&dSG, (size_t)8 * (K >> 5) * 4));
    CK(hipMemset(dX8, 0, (size_t)8 * K));
    CK(hipMemcpy(dA,  hA.data(),  hA.size(),  hipMemcpyHostToDevice));
    CK(hipMemcpy(dAS, hAS.data(), hAS.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dBS, hBS.data(), hBS.size(), hipMemcpyHostToDevice));

    std::vector<void*> dB(rot), dS(rot);                        // nibble + bench q8x planes
    for (int p = 0; p < rot; ++p) {
        std::vector<uint8_t> hB((size_t)N * (K / 2));
        for (auto& v : hB) v = (uint8_t)rnd();
        CK(hipMalloc(&dB[p], hB.size()));
        CK(hipMemcpy(dB[p], hB.data(), hB.size(), hipMemcpyHostToDevice));
        CK(hipMalloc(&dS[p], (size_t)N * rpitch));
    }
    const dim3 rpg((unsigned)std::min<int64_t>(
        ((int64_t)N * ((K >> 1) + (K >> 7)) + 255) / 256, 8192));
    for (int p = 0; p < rot; ++p)                               // repack bench planes
        m4_repack_q8x_kernel<<<rpg, 256>>>((const uint8_t*)dB[p], (const uint8_t*)dBS,
                                           (uint8_t*)dS[p], K / 2, bs_cols, (int)N, (int)K);
    CK(hipDeviceSynchronize());
    const dim3 apg((unsigned)((M * (K >> 5) + 7) / 8));

    // library operand frame (positions follow oGemmNtQ; ASUM null, b_ref absent)
    auto mk = [](void* d, uint32_t dt, int rank, std::vector<int64_t> sh,
                 std::vector<int64_t> stride) {
        RadTensor x{}; x.data = d; x.dtype = dt; x.rank = (uint32_t)rank;
        for (size_t i = 0; i < sh.size(); ++i) { x.shape[i] = sh[i]; x.stride[i] = stride[i]; }
        return x; };
    std::vector<RadTensor> base{
        mk(dA,  RAD_F8E4M3, 2, {M, K}, {K, 1}),
        mk(dAS, RAD_F32, 2, {M, as_cols}, {as_cols, 1}),
        mk(nullptr, RAD_FP4E2M1, 2, {N, K}, {K, 1}),
        mk(dBS, RAD_E8M0, 2, {N, bs_cols}, {bs_cols, 1}),
        mk(dY,  RAD_BF16, 2, {M, N}, {N, 1}),
        mk(nullptr, RAD_F32, 0, {0}, {0}),
        mk(nullptr, RAD_BF16, 0, {0}, {0}),
    };
    RadTensor* t = nullptr;
    CK(hipHostMalloc((void**)&t, sizeof(RadTensor) * 7));
    RadParam pv[] = { RAD_STR("dtype", "mxfp4a8"), RAD_INT("N", N), RAD_INT("K", K),
                      RAD_INT("group", 32) };
    std::vector<uint16_t> ySent((size_t)M * N, kSentinel);
    auto launch_lib = [&](int p) -> int {                       // shipped walk, plugin planes
        std::vector<RadTensor> dv = base;
        dv[2].data = dB[p];
        memcpy(t, dv.data(), sizeof(RadTensor) * 7);
        RadArgs a{}; a.p = pv; a.n_p = 4; a.t = t; a.n_t = 7;
        const int rc = entry(&a, (RadStream)0);
        CK(hipDeviceSynchronize());
        return rc; };
    auto launch_act = [&]() {
        m4_act_i8_kernel<<<apg, 256>>>((const uint8_t*)dA, (int8_t*)dX8, (float*)dSG,
                                       M, (int)K, K); };
#define Q47_LM(WIN_, MT_) \
    q47_lm_kernel<MT_, WIN_><<<(unsigned)N, 32>>>((const uint8_t*)dS[p], (const int8_t*)dX8, \
        (const float*)dSG, (const float*)dAS, (uint16_t*)dY, M, (int)N, (int)K, N, as_cols, 1, 1)
    auto launch_bench = [&](int p, int win) {   // act + walk ONE back-to-back pair, like the
        launch_act();                          // library entry: act is async, ONE sync at end
        if (mt == 4) { if (win == 2) Q47_LM(2, 4); else Q47_LM(1, 4); }
        else         { if (win == 2) Q47_LM(2, 8); else Q47_LM(1, 8); }
        CK(hipDeviceSynchronize()); };
#undef Q47_LM

    printf("LEG r=%d M=%d K=%lld N=%lld rot=%d mt=%d reps=%d plane=%.1fMB regime=%s\n",
           RND, M, K, N, rot, mt, reps, plane_bytes / 1048576.0, regime);

    // ---- warm every arm on every plane (lib repack cache incl.; never in timing) ----
    for (int p = 0; p < rot; ++p) {
        const int rc = launch_lib(p);
        if (rc != 0) { printf("LIB ENTRY REFUSED rc=%d -- shape not serving\n", rc); return 3; }
        launch_bench(p, 1); launch_bench(p, 2);
    }

    // ---- fidelity + EQUIV + row-write gate on plane 0 (sentinel-poisoned) ----
    std::vector<uint16_t> yL((size_t)M * N), yB((size_t)M * N), yW((size_t)M * N);
    CK(hipMemcpy(dY, ySent.data(), (size_t)M * N * 2, hipMemcpyHostToDevice));
    launch_lib(0);
    CK(hipMemcpy(yL.data(), dY, (size_t)M * N * 2, hipMemcpyDeviceToHost));
    CK(hipMemcpy(dY, ySent.data(), (size_t)M * N * 2, hipMemcpyHostToDevice));
    launch_bench(0, 1);
    CK(hipMemcpy(yB.data(), dY, (size_t)M * N * 2, hipMemcpyDeviceToHost));
    CK(hipMemcpy(dY, ySent.data(), (size_t)M * N * 2, hipMemcpyHostToDevice));
    launch_bench(0, 2);
    CK(hipMemcpy(yW.data(), dY, (size_t)M * N * 2, hipMemcpyDeviceToHost));
    long eqLB = 0, eqBW = 0, eqLW = 0;
    for (size_t i = 0; i < yL.size(); ++i) {
        eqLB += (yL[i] == yB[i]); eqBW += (yB[i] == yW[i]); eqLW += (yL[i] == yW[i]);
    }
    printf("FIDELITY r=%d M=%d K=%lld N=%lld base_vs_lib BITWISE %ld/%zu yhash L=%016llx B=%016llx %s\n",
           RND, M, K, N, eqLB, yL.size(), (unsigned long long)fnv1a(yL),
           (unsigned long long)fnv1a(yB), eqLB == (long)yL.size() ? "PASS" : "FAIL");
    printf("EQUIV    r=%d M=%d K=%lld N=%lld win_vs_base BITWISE %ld/%zu win_vs_lib %ld/%zu yhash W=%016llx %s\n",
           RND, M, K, N, eqBW, yL.size(), eqLW, yL.size(),
           (unsigned long long)fnv1a(yW),
           eqBW == (long)yL.size() ? "PASS" : "FAIL");
    {   // row-write counts per arm: a row counts WRITTEN if it is not all-sentinel
        auto rows = [&](const std::vector<uint16_t>& y) {
            long w = 0;
            for (int m = 0; m < M; ++m) {
                bool any = false;
                for (long long c = 0; c < N; ++c) any |= (y[(size_t)m * N + c] != kSentinel);
                w += any;
            }
            return w; };
        const long rl = rows(yL), rb = rows(yB), rw = rows(yW);
        printf("ROWWRITE r=%d M=%d expected_live=%d lib=%ld base=%ld win=%ld %s\n",
               RND, M, live, rl, rb, rw,
               (rl == live && rb == live && rw == live) ? "PASS" : "FAIL");
    }

    // ---- timing: clocks first, then FINE-GRAINED arm round-robins ----
    // W66 law (this session): 24-rep BLOCKS of a 0.1ms kernel (≈3ms) are shorter
    // than the DVFS ramp -- the measured pass showed a ±20% IN-PROCESS POSITION
    // effect (first arm slow, last arm fast) that crossed rounds. Fix: (1) a
    // 400-launch pre-burn over every plane/arm before any timing; (2) 9 cycles of
    // 3 arms x REPS reps each, the arm ORDER ROTATED every cycle, so each arm
    // occupies every position equally; per-arm stats = median/min/max over the
    // 9 cycle samples. REV rotates the cycle start across rounds as before.
    for (int i = 0; i < 400; ++i) {                             // clock pre-burn
        const int p = i % rot;
        const int w = (i / rot) % 3;
        if (w == 0) launch_lib(p); else launch_bench(p, w);     // w=1 base, w=2 win
    }
    auto timeit = [&](int which, double* med, double* lo, double* hi, int reps_) {
        std::vector<double> ts;                                 // 0 lib 1 base 2 win
        for (int r = 0; r < reps_; ++r) {
            const int p = r % rot;
            timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            if (which == 0) launch_lib(p);
            else launch_bench(p, which == 2 ? 2 : 1);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            ts.push_back((t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec));
        }
        std::sort(ts.begin(), ts.end());
        *med = ts[ts.size() / 2]; *lo = ts.front(); *hi = ts.back(); };
    int seq[3] = { 0, 1, 2 };
    if (REV) std::reverse(seq, seq + 3);
    const char* aname[3] = { "lib", "base", "win" };
    std::vector<uint16_t> yT((size_t)M * N);
    double samp[3][9];
    for (int cyc = 0; cyc < 9; ++cyc) {                         // rotated round-robins
        const int s0 = cyc % 3;
        for (int j = 0; j < 3; ++j) {
            const int idx = seq[(j + s0) % 3];
            double med, lo, hi;
            timeit(idx, &med, &lo, &hi, reps);
            samp[idx][cyc] = med;
        }
    }
    for (int idx = 0; idx < 3; ++idx) {
        std::vector<double> v(samp[idx], samp[idx] + 9);
        std::sort(v.begin(), v.end());
        CK(hipMemcpy(yT.data(), dY, (size_t)M * N * 2, hipMemcpyDeviceToHost));
        printf("AB r=%d arm=%s ROW M=%d K=%lld N=%lld rot=%d reps=%d | t_med=%.3f ms"
               "  q8x %.1f GB/s nonzero=%llu/%llu\n",
               RND, aname[idx], M, K, N, rot, reps * 9, v[4] * 1e3,
               plane_bytes / v[4] / 1e9,
               (unsigned long long)nz16(yT), (unsigned long long)((size_t)M * N));
        printf("   %s cyc-spread %.3f..%.3f ms\n", aname[idx], v.front() * 1e3, v.back() * 1e3);
        fflush(stdout);
    }
    return 0;
}
