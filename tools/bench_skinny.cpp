// bench_skinny.cpp -- Q21 A/B: WMMA skinny decode GEMM vs the shipped w2a8 GEMV.
//
// Donor pattern (LIBR35D_TRIAGE delta5, r35d_8a8_narrow_kernel): 16-row fragment tile
// as matrix_a, tokens as matrix_b, i32 WMMA accumulation. On gfx1100 the probe
// (probe_wmma.cpp, window 17) measured i8 WMMA at 3.31x the dot4 chain and showed the
// M=14 GEMV row runs at 10.6% of even the dot4 peak -- staging-bound, and WMMA needs
// 4x less LDS bytes per MAC and NO cross-lane flush (each lane's accumulator registers
// ARE output elements: row 2k+(l>>4), col l&15).
//
// Bit-identity by construction: per 128-group the i32 fragment sum is the same exact
// integer as the GEMV's group sum (integer addition is order-free); the float epilogue
// ((is - z*SUM)*AS)*sb is applied per group in ascending group order, verbatim. The
// w2 pairing contract is honored: code slot s multiplies activation column
// w2_step_pos(s) (the bits-2/3 swap never crosses a 128 boundary, so one 16-slot WMMA
// k-tile lives inside one scale group).
//
// Arms in one process, same operands:
//   ROW    = dlopen'd library entry (the shipped GEMV)
//   NAIVE  = scalar oracle (independent re-derivation of the contract)
//   SKINNY = this prototype WMMA kernel
// Build/run (radiance-dev, card):
//   hipcc -O2 --offload-arch=gfx1100 -mno-wavefrontsize64 -std=c++17 \
//         -I/opt/radiance/include/radiance -w bench_skinny.cpp -o /s/sk -ldl
//   ROCR_VISIBLE_DEVICES=1 LIBR3D_SO=... /s/sk <M> <K> <N> <REPS> <ROT>
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <vector>
#include <rocwmma/rocwmma.hpp>
#include "rad_abi.h"
#include "rad_types.h"
#include "rad_planes.h"

namespace roCW = rocwmma;
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP %s: %s\n", #x, hipGetErrorName(e_)); exit(2); } } while (0)

// ---- contract helpers, verbatim from src/r3d_gemm_w2a8.hip ------------------
__device__ __forceinline__ int w2_step_pos(int k) {
    return (k & ~12) | ((k & 4) << 1) | ((k & 8) >> 1);
}
__device__ __forceinline__ uint32_t w2_expand4(uint32_t b) {
    return (b & 3u) | ((b >> 2 & 3u) << 8) | ((b >> 4 & 3u) << 16) | ((b >> 6) << 24);
}
__device__ __forceinline__ uint4 w2_expand16(uint32_t w) {
    uint4 o;
    o.x = w2_expand4(w & 0xffu); o.y = w2_expand4((w >> 8) & 0xffu);
    o.z = w2_expand4((w >> 16) & 0xffu); o.w = w2_expand4(w >> 24);
    return o;
}
__device__ __forceinline__ uint16_t w2_f32_to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
constexpr int kLdm = 136;   // LDS stride, pad 8 to break the 128B full-conflict row

// ---- SKINNY: 16-row WMMA tile per wave, tokens as matrix_b ------------------
__global__ __launch_bounds__(256) void skinny_kernel(
    const int8_t* __restrict__ A, const uint8_t* __restrict__ Bp,
    uint16_t* __restrict__ Y, const float* __restrict__ AS,
    const __half* __restrict__ SC, const uint8_t* __restrict__ ZP,
    const float* __restrict__ SUM, int M, int N, int K, long ngrp)
{
    __shared__ __align__(16) int8_t  xs[16 * kLdm];            // tokens x slots
    __shared__ __align__(16) int8_t  ws[8][16 * kLdm];         // per-wave rows x slots
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tile = (blockIdx.x * 8 + w) * 16;                // first row of tile
    if (tile >= N) return;
    // i32-accumulator lane map, PROBE-PINNED this window (probe_wmma imaps, i8 x i8):
    // lane l holds row 2k + (l>>4), col l&15 -- IDENTICAL to the f32 accumulator map.
    // (An intermediate read of the probe wrongly saw lane pairs; that misread, not the
    // hardware, produced the first mismatching build. The imaps table is the record.)
    const int tok = lane & 15;
    roCW::fragment<roCW::matrix_a, 16, 16, 16, int8_t, roCW::row_major> fa;   // i8 x i8: the probe's live 3.31x stage
    roCW::fragment<roCW::matrix_b, 16, 16, 16, int8_t, roCW::col_major> fb;
    roCW::fragment<roCW::accumulator, 16, 16, 16, int32_t> fc;
    float y[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) y[k] = 0.0f;

    for (long g = 0; g < ngrp; ++g) {
        const long k0 = g * 128;
        for (int i = threadIdx.x; i < 16 * 128; i += 256) {    // stage tokens
            const int m = i >> 7, s = i & 127;
            xs[m * kLdm + s] = (m < M) ? A[(int64_t)m * K + k0 + w2_step_pos(s)] : 0;
        }
        {   // stage 16 rows x 8 dwords -- EVERY WAVE stages its OWN tile: the first
            // build gated this on tid<128, which is waves 0..3 only, and indexed the
            // destination by the writing thread's wave id -- so ws[w] got a quarter of
            // its rows and waves 4..7 read uninitialized LDS (the ledger's dead-row
            // family, this time on the LDS side: near-miss values, not garbage).
            const uint32_t* bp32 = (const uint32_t*)Bp;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int idx = q * 32 + lane;              // 0..127 within the wave
                const int r = idx >> 3, d = idx & 7, rr = tile + r;
                uint4 e = make_uint4(0, 0, 0, 0);
                if (rr < N)
                    e = w2_expand16(bp32[(int64_t)rr * (K >> 4) + (k0 >> 4) + d]);
                *(uint4*)&ws[w][r * kLdm + d * 16] = e;     // codes 0..3: sign-extension-free i8
            }
        }
        __syncthreads();
        roCW::fill_fragment(fc, (int32_t)0);
#pragma unroll
        for (int kk = 0; kk < 128; kk += 16) {
            roCW::load_matrix_sync(fa, (const int8_t*)&ws[w][kk], kLdm);
            roCW::load_matrix_sync(fb, (const int8_t*)&xs[kk], kLdm);
            roCW::mma_sync(fc, fa, fb, fc);
        }
        if (tok < M) {
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int rr = tile + 2 * k + (lane >> 4);     // pinned lane map (i32)
                if (rr >= N) continue;
                const float sb = __half2float(SC[(int64_t)rr * ngrp + g]);
                const float term = (float)ZP[(int64_t)rr * ngrp + g] * SUM[g * 16 + tok];
                y[k] += (((float)fc.x[k] - term) * AS[tok]) * sb;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int rr = tile + 2 * k + (lane >> 4);
        if (tok < M && rr < N) Y[(int64_t)tok * N + rr] = w2_f32_to_bf16(y[k]);
    }
}

// ---- NAIVE oracle: one thread per (row, token), same contract, no WMMA ------
__global__ void naive_kernel(
    const int8_t* __restrict__ A, const uint8_t* __restrict__ Bp,
    uint16_t* __restrict__ Y, const float* __restrict__ AS,
    const __half* __restrict__ SC, const uint8_t* __restrict__ ZP,
    const float* __restrict__ SUM, int M, int N, int K, long ngrp)
{
    const int r = blockIdx.x * 256 + threadIdx.x;
    if (r >= N) return;
    for (int t = 0; t < M; ++t) {
        float acc = 0.0f;
        for (long g = 0; g < ngrp; ++g) {
            int32_t part = 0;
            for (int s = 0; s < 128; ++s) {
                const int col = (int)(g * 128) + w2_step_pos(s);
                const int code = (Bp[(int64_t)r * (K >> 2) + ((g * 128 + s) >> 2)]
                                  >> (2 * ((g * 128 + s) & 3))) & 3;
                part += (int)A[(int64_t)t * K + col] * code;
            }
            const float sb = __half2float(SC[(int64_t)r * ngrp + g]);
            const float term = (float)ZP[(int64_t)r * ngrp + g] * SUM[g * 16 + t];
            acc += (((float)part - term) * AS[t]) * sb;
        }
        Y[(int64_t)t * N + r] = w2_f32_to_bf16(acc);
    }
}

// ---- fixture: identical construction to bench_w2a8.cpp -----------------------
static uint16_t f32_to_f16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    int32_t e = (int32_t)((u >> 23) & 0xff) - 127 + 15;
    uint32_t s = (u >> 16) & 0x8000u;
    if (e <= 0) return (uint16_t)s;
    if (e >= 31) return (uint16_t)(s | 0x7c00u);
    uint32_t mant = u & 0x7fffffu;
    uint16_t h = (uint16_t)(s | ((uint32_t)e << 10) | (mant >> 13));
    if (mant & 0x1000u) ++h;
    return h;
}

int main(int argc, char** argv) {
    const int      M    = argc > 1 ? atoi(argv[1]) : 14;
    const long long K   = argc > 2 ? atoll(argv[2]) : 16384;
    const long long N   = argc > 3 ? atoll(argv[3]) : 16384;
    const int      reps = argc > 4 ? atoi(argv[4]) : 30;
    const int      rot  = argc > 5 ? atoi(argv[5]) : 1;
    if (M < 1 || M > 16 || K <= 0 || N <= 0 || K % 128 || rot < 1 || rot > 4) {
        printf("usage: bench_skinny M(1..16) K(%%128) N REPS ROT(<=4)\n"); return 2; }

    void* dl = dlopen(getenv("LIBR3D_SO") ? getenv("LIBR3D_SO") : "libr3d.so", RTLD_NOW | RTLD_LOCAL);
    if (!dl) { printf("dlopen libr3d: %s\n", dlerror()); return 1; }
    auto entry = (int (*)(const RadArgs*, RadStream))dlsym(dl, "r3d_rad_gemm_nt_q_w2a8_decode");
    if (!entry) { printf("dlsym entry: %s\n", dlerror()); return 1; }

    const long long ngrp = K / 128, mpad = 16;
    uint32_t st = 0x2468ace1u;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };

    std::vector<int8_t> hA(16 * K);
    for (auto& v : hA) v = (int8_t)((rnd() % 8) - 4);
    std::vector<float> hAS(16);
    for (auto& v : hAS) v = 0.01f + 0.001f * (float)(rnd() % 16);
    std::vector<float> hSUM(ngrp * mpad, 0.f);
    for (long long m = 0; m < 16; ++m)
        for (long long g = 0; g < ngrp; ++g) {
            float s = 0;
            for (long long k = g * 128; k < (g + 1) * 128; ++k) s += (float)hA[m * K + k];
            hSUM[g * mpad + m] = s;
        }
    const int64_t sc_bytes = 2 * ngrp * N, zp_bytes = ngrp * N;
    const int64_t sc_off = rad_planes_round((int64_t)(sizeof(RadPlanesHeader) +
                                                      2 * sizeof(RadPlanesEntry)));
    const int64_t zp_off = rad_planes_round(sc_off + sc_bytes);
    const int64_t blob   = zp_off + zp_bytes;
    std::vector<uint8_t> scBlob((size_t)blob, 0);
    {
        auto* h = (RadPlanesHeader*)scBlob.data();
        h->magic = RAD_PLANES_MAGIC; h->n_planes = 2; h->transform[0] = 0; h->bytes = blob;
        auto* e = (RadPlanesEntry*)(scBlob.data() + sizeof(RadPlanesHeader));
        std::strcpy(e[0].role, "scale"); e[0].dtype = RAD_F16; e[0].kind = RAD_PLANE_TILED;
        e[0].block[0] = 1; e[0].block[1] = 128; e[0].rows = N; e[0].cols = ngrp;
        e[0].row_bytes = 2 * ngrp; e[0].offset = sc_off;
        std::strcpy(e[1].role, "zero");  e[1].dtype = RAD_U8;  e[1].kind = RAD_PLANE_TILED;
        e[1].block[0] = 1; e[1].block[1] = 128; e[1].rows = N; e[1].cols = ngrp;
        e[1].row_bytes = ngrp; e[1].offset = zp_off;
        for (long long r = 0; r < N; ++r)
            for (long long g = 0; g < ngrp; ++g) {
                uint16_t h16 = f32_to_f16(0.0030f + 0.000125f * (float)(rnd() % 32));
                memcpy(&scBlob[(size_t)(sc_off + (r * ngrp + g) * 2)], &h16, 2);
                scBlob[(size_t)(zp_off + r * ngrp + g)] = (uint8_t)(rnd() % 3);
            }
    }
    void *dAS, *dSUM, *dBlob, *dY, *hY_p;
    CK(hipMalloc(&dAS, hAS.size() * 4));
    CK(hipMalloc(&dSUM, hSUM.size() * 4));
    CK(hipMalloc(&dBlob, scBlob.size()));
    CK(hipMalloc(&dY, (size_t)16 * N * 2));
    CK(hipHostMalloc(&hY_p, (size_t)16 * N * 2));
    CK(hipMemcpy(dAS, hAS.data(), hAS.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dSUM, hSUM.data(), hSUM.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dBlob, scBlob.data(), scBlob.size(), hipMemcpyHostToDevice));
    const __half* SC = (const __half*)(scBlob.data() + sc_off);   // device-side recompute:
    void* dSC; CK(hipMalloc(&dSC, sc_bytes));                     // pass raw scale/zp too
    CK(hipMemcpy(dSC, scBlob.data() + sc_off, sc_bytes, hipMemcpyHostToDevice));
    const uint8_t* ZP = (const uint8_t*)(scBlob.data() + zp_off);
    void* dZP; CK(hipMalloc(&dZP, zp_bytes));
    CK(hipMemcpy(dZP, scBlob.data() + zp_off, zp_bytes, hipMemcpyHostToDevice));

    std::vector<uint8_t> hB((size_t)N * (K / 4));
    std::vector<void*> dB(rot), dA(rot);
    for (int p = 0; p < rot; ++p) {
        for (auto& v : hB) v = (uint8_t)rnd();
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
        mk(nullptr, RAD_I8, 2, {16, K}, {K, 1}),
        mk(dAS, RAD_F32, 1, {16}, {1}),
        mk(nullptr, RAD_U2, 2, {N, K}, {K, 1}),
        mk(dBlob, RAD_U8, 1, {blob}, {1}),
        mk(dY, RAD_BF16, 2, {M, N}, {N, 1}),
        mk(dSUM, RAD_F32, 2, {ngrp, mpad}, {mpad, 1}),
    };
    RadTensor* t = nullptr;
    CK(hipHostMalloc((void**)&t, sizeof(RadTensor) * 6));
    RadParam pv[] = { RAD_STR("dtype", "w2a8"), RAD_INT("N", N), RAD_INT("K", K),
                      RAD_INT("group", 128) };
    auto launch_row = [&](int p) {
        std::vector<RadTensor> dv = base;
        dv[0].data = dA[p]; dv[2].data = dB[p];
        memcpy(t, dv.data(), sizeof(RadTensor) * 6);
        RadArgs a{}; a.p = pv; a.n_p = 4; a.t = t; a.n_t = 6;
        const int rc = entry(&a, (RadStream)0);
        CK(hipDeviceSynchronize());
        return rc; };
    auto launch_skinny = [&](int p) {
        skinny_kernel<<<(unsigned)((N + 127) / 128), 256>>>(
            (const int8_t*)dA[p], (const uint8_t*)dB[p], (uint16_t*)dY,
            (const float*)dAS, (const __half*)dSC, (const uint8_t*)dZP,
            (const float*)dSUM, M, (int)N, (int)K, ngrp);
        CK(hipDeviceSynchronize()); };

    // correctness at plane 0: ROW vs SKINNY vs NAIVE, bitwise on the bf16 outputs
    std::vector<uint16_t> yRow((size_t)M * N), ySk((size_t)M * N), yNv((size_t)M * N);
    if (launch_row(0) != 0) { printf("ENTRY REFUSED rc -- row not serving this shape\n"); return 3; }
    CK(hipMemcpy(yRow.data(), dY, (size_t)M * N * 2, hipMemcpyDeviceToHost));
    CK(hipMemset(dY, 0, (size_t)16 * N * 2));
    launch_skinny(0);
    CK(hipMemcpy(ySk.data(), dY, (size_t)M * N * 2, hipMemcpyDeviceToHost));
    CK(hipMemset(dY, 0, (size_t)16 * N * 2));
    naive_kernel<<<(unsigned)((N + 255) / 256), 256>>>(
        (const int8_t*)dA[0], (const uint8_t*)dB[0], (uint16_t*)dY,
        (const float*)dAS, (const __half*)dSC, (const uint8_t*)dZP,
        (const float*)dSUM, M, (int)N, (int)K, ngrp);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(yNv.data(), dY, (size_t)M * N * 2, hipMemcpyDeviceToHost));
    long sk_match = 0, nv_match = 0, sk_nv = 0;
    for (size_t i = 0; i < yRow.size(); ++i) {
        sk_match += (ySk[i] == yRow[i]);
        nv_match += (yNv[i] == yRow[i]);
        sk_nv    += (ySk[i] == yNv[i]);
    }
    printf("CHECK M=%d K=%lld N=%lld: SKINNY==ROW %ld/%zu  NAIVE==ROW %ld/%zu  SKINNY==NAIVE %ld/%zu\n",
           M, K, N, sk_match, yRow.size(), nv_match, yRow.size(), sk_nv, yRow.size());
    if (sk_match != (long)yRow.size() || nv_match != (long)yRow.size()) {
        long shown = 0;
        for (size_t i = 0; i < yRow.size() && shown < 6; ++i)
            if (ySk[i] != yRow[i]) {
                printf("  [t=%zu r=%zu] ROW=%04x SKINNY=%04x NAIVE=%04x\n",
                       i / (size_t)N, i % (size_t)N, yRow[i], ySk[i], yNv[i]); ++shown; }
    }

    // alternating A/B timing (card protocol): A=ROW, B=SKINNY, same planes, same reps
    for (int p = 0; p < rot; ++p) { launch_row(p); launch_skinny(p); }   // warm + repacks
    std::vector<double> ta, tb;
    for (int r = 0; r < reps; ++r) {
        timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0); launch_row(r % rot);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ta.push_back((t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec));
        clock_gettime(CLOCK_MONOTONIC, &t0); launch_skinny(r % rot);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        tb.push_back((t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec));
    }
    std::sort(ta.begin(), ta.end()); std::sort(tb.begin(), tb.end());
    const double med = ta[ta.size() / 2], meds = tb[tb.size() / 2];
    printf("A ROW    M=%d K=%lld N=%lld rot=%d | t_med=%.3f ms  p2=%.3f p8=%.3f  u2-stream %.1f GB/s\n",
           M, K, N, rot, med * 1e3, ta[ta.size() / 5] * 1e3, ta[4 * ta.size() / 5] * 1e3,
           (double)N * K / 4.0 / med / 1e9);
    printf("B SKINNY M=%d K=%lld N=%lld rot=%d | t_med=%.3f ms  p2=%.3f p8=%.3f  u2-stream %.1f GB/s\n",
           M, K, N, rot, meds * 1e3, tb[tb.size() / 5] * 1e3, tb[4 * tb.size() / 5] * 1e3,
           (double)N * K / 4.0 / meds / 1e9);
    printf("DELTA    M=%d | (A/B-1)*100 = %+.1f%%\n", M, (med / meds - 1.0) * 100.0);
    return 0;
}
