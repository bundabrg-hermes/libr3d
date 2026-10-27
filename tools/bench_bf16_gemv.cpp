// bench_bf16_gemv.cpp -- steady-state bench for the bf16 gemm_nt GEMV row (Q33).
// Drives the shipped entry r3d_rad_gemm_nt at M = 1 (the alignment-gated GEMV
// band) with random operands, mirroring tools/bench_w2a8.cpp's protocol:
//   * L2 law: the serve streams whole planes per token; the bf16 row streams
//     N*K*2 bytes per launch, so shapes at N*K*2 >> 96 MB measure the serving
//     regime; ROT > 1 rotates distinct B planes.
//   * Liveness law: random operands + nonzero-output count printed with the
//     rate -- a rate from a dead row is a lie. The dpp=<N> liveness line from
//     R3D_DEBUG_GEMM must name the arm each leg actually rode.
// Build/run (in radiance-dev ON the card, gfx1100 pinned via ROCR_VISIBLE_DEVICES):
//   hipcc -O1 -std=c++20 -I/opt/radiance/include/radiance -w tools/bench_bf16_gemv.cpp \
//     -o /tmp/bbf -ldl
//   ROCR_VISIBLE_DEVICES=1 LIBR3D_SO=$PWD/build_v152/radiance_home/kernels/libr3d.so \
//     R3D_DEBUG_GEMM=1 /tmp/bbf <K> <N> <REPS> [ROT]
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

static void die(const char* m, hipError_t e) { if (e == hipSuccess) return;
    fprintf(stderr, "%s: %s\n", m, hipGetErrorName(e)); exit(2); }

int main(int argc, char** argv) {
    const long long K    = argc > 1 ? atoll(argv[1]) : 16384;
    const long long N    = argc > 2 ? atoll(argv[2]) : 16384;
    const int       reps = argc > 3 ? atoi(argv[3]) : 30;
    const int       rot  = argc > 4 ? atoi(argv[4]) : 1;
    const long long M = 1;   /* the GEMV band is M == 1 by dispatch law */
    if (K <= 0 || N <= 0 || K % 8 || rot < 1 || rot > 4) {
        printf("usage: bench_bf16_gemv K(%8) N REPS ROT(<=4)\n"); return 2; }

    void* dl = dlopen(getenv("LIBR3D_SO") ? getenv("LIBR3D_SO")
                                          : "/build/radiance_home/kernels/libr3d.so",
                      RTLD_NOW | RTLD_LOCAL);
    if (!dl) { printf("dlopen libr3d: %s\n", dlerror()); return 1; }
    auto entry = (int (*)(const RadArgs*, RadStream))dlsym(dl, "r3d_rad_gemm_nt");
    if (!entry) { printf("dlsym entry: %s\n", dlerror()); return 1; }

    hipDeviceProp_t prop;
    die("getprops", hipGetDeviceProperties(&prop, 0));
    printf("DEV name=%s arch=%s l2Bytes=%zu memGB=%.1f\n", prop.name, prop.gcnArchName,
           prop.l2CacheSize, prop.totalGlobalMem / 1073741824.0);

    uint32_t st = 0x2468ace1u;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };
    // bf16 in [1,2) x sign: exponent 0x7F fixed, 7 random mantissa bits -- no
    // inf/nan class, sums of K terms stay far from f32 overflow.
    auto rbf = [&]() { return (uint16_t)((rnd() & 0x8000u) | 0x3F80u | (rnd() & 0x7fu)); };

    std::vector<uint16_t> hA((size_t)K);
    for (auto& v : hA) v = rbf();

    void *dA, *dY, *hY_p;
    die("mallocA", hipMalloc((void**)&dA, hA.size() * 2));
    die("mallocY", hipMalloc((void**)&dY, (size_t)M * N * 2));
    die("mallocYh", hipHostMalloc(&hY_p, (size_t)M * N * 2));
    die("copyA", hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));

    std::vector<void*> dB(rot);
    for (int p = 0; p < rot; ++p) {
        std::vector<uint16_t> hB((size_t)N * K);
        for (auto& v : hB) v = rbf();
        die("mallocB", hipMalloc((void**)&dB[p], hB.size() * 2));
        die("copyB",  hipMemcpy(dB[p], hB.data(), hB.size() * 2, hipMemcpyHostToDevice));
    }

    auto mk = [](void* d, uint32_t dt, int rank, std::vector<int64_t> sh,
                 std::vector<int64_t> stride) {
        RadTensor x{}; x.data = d; x.dtype = dt; x.rank = (uint32_t)rank;
        for (size_t i = 0; i < sh.size(); ++i) { x.shape[i] = sh[i]; x.stride[i] = stride[i]; }
        return x; };

    std::vector<RadTensor> base{
        mk(dA, RAD_BF16, 2, {M, K}, {K, 1}),
        mk(nullptr, RAD_BF16, 2, {N, K}, {K, 1}),
        mk(dY, RAD_BF16, 2, {M, N}, {N, 1}),
    };
    RadTensor* t = nullptr;
    die("mallocT", hipHostMalloc((void**)&t, sizeof(RadTensor) * 3));
    RadParam pv[] = { RAD_INT("M", M), RAD_INT("N", N), RAD_INT("K", K) };

    auto launch = [&](int p) {
        std::vector<RadTensor> dv = base;
        dv[1].data = dB[p];
        memcpy(t, dv.data(), sizeof(RadTensor) * 3);
        RadArgs a{}; a.p = pv; a.n_p = 3; a.t = t; a.n_t = 3;
        const int rc = entry(&a, (RadStream)0);
        die("launch", hipDeviceSynchronize());
        return rc;
    };

    if (launch(0) != 0) { printf("ENTRY REFUSED rc -- row not serving this shape\n"); return 3; }

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
    const double gbs = (double)N * (double)K * 2.0 / med / 1e9;   // bf16 plane bytes

    die("copyY", hipMemcpy(hY_p, dY, (size_t)M * N * 2, hipMemcpyDeviceToHost));
    size_t nz = 0;
    {
        std::vector<uint16_t> y((size_t)M * N);
        memcpy(y.data(), hY_p, y.size() * 2);
        for (auto v : y) if (v) ++nz;
    }

    printf("ROW M=%lld K=%lld N=%lld rot=%d reps=%d | t_med=%.3f ms  bf16-stream %.1f GB/s"
           "  nonzero=%zu/%lld %s\n",
           M, K, N, rot, reps, med * 1e3, gbs, nz, (long long)(M * N),
           nz ? "" : " <-- DEAD ROW, RATE IS A LIE");
    return nz ? 0 : 4;
}
