// bench_w2a8.cpp -- steady-state bench for the w2a8 decode GEMV row (Q18).
// Forks tools/repro_w2a8.hip's operand construction (the joined scale,zero blob the
// kbench recorders cannot describe) and times the row entry at serving-shape sizes.
//
// L2 law (adopted card protocol): the serve streams whole planes per token; replay of
// one small plane measures a regime that never runs. The q8 arm streams N*K BYTES per
// launch (the repacked plane), so a single plane at N*K >= 96 MB is already past L2;
// ROT > 1 rotates distinct B planes (the plugin's repack cache holds 4 -- ROT <= 4).
//
// Liveness law (the all-zero-fixture lesson): random operands everywhere and a
// nonzero-output count printed with the rate -- a rate from a dead row is a lie.
//
// Build/run (in radiance-dev ON the card, gfx1100 pinned via ROCR_VISIBLE_DEVICES):
//   hipcc -O1 -std=c++20 -I/opt/radiance/include/radiance -w tools/bench_w2a8.cpp \
//     -o /tmp/bw2 -ldl
//   ROCR_VISIBLE_DEVICES=1 LIBR3D_SO=$PWD/build_v126/radiance_home/kernels/libr3d.so \
//     R3D_DEBUG_GEMM=1 /tmp/bw2 <M> <K> <N> <REPS> [ROT]
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
#include "rad_planes.h"

static void die(const char* m, hipError_t e) { if (e == hipSuccess) return;
    fprintf(stderr, "%s: %s\n", m, hipGetErrorName(e)); exit(2); }

static uint16_t f32_to_f16(float f) {   // exact for the dyadic scales used here
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
    const int      M    = argc > 1 ? atoi(argv[1]) : 1;
    const long long K   = argc > 2 ? atoll(argv[2]) : 16384;
    const long long N   = argc > 3 ? atoll(argv[3]) : 16384;
    const int      reps = argc > 4 ? atoi(argv[4]) : 30;
    const int      rot  = argc > 5 ? atoi(argv[5]) : 1;
    if (M < 1 || M > 16 || K <= 0 || N <= 0 || K % 128 || rot < 1 || rot > 4) {
        printf("usage: bench_w2a8 M(1..16) K(%128) N REPS ROT(<=4)\n"); return 2; }

    void* dl = dlopen(getenv("LIBR3D_SO") ? getenv("LIBR3D_SO")
                                          : "/build/radiance_home/kernels/libr3d.so",
                      RTLD_NOW | RTLD_LOCAL);
    if (!dl) { printf("dlopen libr3d: %s\n", dlerror()); return 1; }
    auto entry = (int (*)(const RadArgs*, RadStream))
                     dlsym(dl, "r3d_rad_gemm_nt_q_w2a8_decode");
    if (!entry) { printf("dlsym entry: %s\n", dlerror()); return 1; }

    hipDeviceProp_t prop;
    die("getprops", hipGetDeviceProperties(&prop, 0));
    printf("DEV name=%s arch=%s l2Bytes=%zu memGB=%.1f\n", prop.name, prop.gcnArchName,
           prop.l2CacheSize, prop.totalGlobalMem / 1073741824.0);

    const long long ngrp = K / 128, mpad = 16;
    uint32_t st = 0x2468ace1u;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };

    std::vector<int8_t> hA(16 * K);
    for (auto& v : hA) v = (int8_t)((rnd() % 8) - 4);
    std::vector<float>  hAS(16);
    for (auto& v : hAS) v = 0.01f + 0.001f * (float)(rnd() % 16);
    std::vector<float>  hSUM(ngrp * mpad, 0.f);
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
    die("mallocAS",  hipMalloc((void**)&dAS,  hAS.size() * 4));
    die("mallocSUM", hipMalloc((void**)&dSUM, hSUM.size() * 4));
    die("mallocBlob", hipMalloc((void**)&dBlob, scBlob.size()));
    die("mallocY",   hipMalloc((void**)&dY,   (size_t)16 * N * 2));
    die("mallocYh",  hipHostMalloc(&hY_p, (size_t)16 * N * 2));
    die("copyAS",  hipMemcpy(dAS,  hAS.data(),  hAS.size() * 4,  hipMemcpyHostToDevice));
    die("copySUM", hipMemcpy(dSUM, hSUM.data(), hSUM.size() * 4, hipMemcpyHostToDevice));
    die("copyBlob", hipMemcpy(dBlob, scBlob.data(), scBlob.size(), hipMemcpyHostToDevice));

    std::vector<uint8_t> hB((size_t)N * (K / 4));
    std::vector<void*> dB(rot), dA(rot);
    for (int p = 0; p < rot; ++p) {
        for (auto& v : hB) v = (uint8_t)rnd();                      // four u2 codes a byte
        die("mallocB", hipMalloc((void**)&dB[p], hB.size()));
        die("copyB",   hipMemcpy(dB[p], hB.data(), hB.size(), hipMemcpyHostToDevice));
        if (p == 0) {
            die("mallocA", hipMalloc((void**)&dA[0], hA.size()));
            die("copyA",   hipMemcpy(dA[0], hA.data(), hA.size(), hipMemcpyHostToDevice));
        }
        for (int q = 1; q < rot; ++q) dA[q] = dA[0];                // one activation plane
    }

    auto mk = [](void* d, uint32_t dt, int rank, std::vector<int64_t> sh,
                 std::vector<int64_t> stride) {
        RadTensor x{}; x.data = d; x.dtype = dt; x.rank = (uint32_t)rank;
        for (size_t i = 0; i < sh.size(); ++i) { x.shape[i] = sh[i]; x.stride[i] = stride[i]; }
        return x; };

    std::vector<RadTensor> base{
        mk(nullptr, RAD_I8, 2,  {16, K}, {K, 1}),
        mk(dAS,  RAD_F32, 1, {16}, {1}),
        mk(nullptr, RAD_U2, 2,  {N, K}, {K, 1}),
        mk(dBlob, RAD_U8, 1, {blob}, {1}),
        mk(dY,   RAD_BF16, 2, {M, N}, {N, 1}),
        mk(dSUM, RAD_F32, 2, {ngrp, mpad}, {mpad, 1}),
    };
    RadTensor* t = nullptr;
    die("mallocT", hipHostMalloc((void**)&t, sizeof(RadTensor) * 6));
    RadParam pv[] = { RAD_STR("dtype", "w2a8"), RAD_INT("N", N), RAD_INT("K", K),
                      RAD_INT("group", 128) };

    auto launch = [&](int p) {
        std::vector<RadTensor> dv = base;
        dv[0].data = dA[p]; dv[2].data = dB[p];
        memcpy(t, dv.data(), sizeof(RadTensor) * 6);
        RadArgs a{}; a.p = pv; a.n_p = 4; a.t = t; a.n_t = 6;
        const int rc = entry(&a, (RadStream)0);
        die("launch", hipDeviceSynchronize());
        return rc;
    };

    // Warmup: fill the repack cache for every plane (the once-per-process repack runs
    // on the plugin's own stream with a host sync -- it must not land in the timing).
    for (int p = 0; p < rot; ++p)
        if (launch(p) != 0) { printf("ENTRY REFUSED rc -- row not serving this shape\n"); return 3; }

    std::vector<double> tms;
    for (int r = 0; r < reps; ++r) {
        timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        launch(r % rot);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        tms.push_back((t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec));
    }
    std::sort(tms.begin(), tms.end());
    const double med = tms[tms.size() / 2];
    // The row streams the repacked plane: N*K bytes per launch (q8 codes, pitch kept).
    const double gbs = (double)N * (double)K / med / 1e9;

    die("copyY", hipMemcpy(hY_p, dY, (size_t)M * N * 2, hipMemcpyDeviceToHost));
    size_t nz = 0;
    {
        std::vector<uint16_t> y((size_t)M * N);
        memcpy(y.data(), hY_p, y.size() * 2);
        for (auto v : y) if (v) ++nz;
    }

    printf("ROW M=%d K=%lld N=%lld rot=%d reps=%d | t_med=%.3f ms  q8-stream %.1f GB/s"
           "  nonzero=%zu/%lld %s\n",
           M, K, N, rot, reps, med * 1e3, gbs, nz, (long long)(M * N),
           nz ? "" : " <-- DEAD ROW, RATE IS A LIE");
    return nz ? 0 : 4;
}
