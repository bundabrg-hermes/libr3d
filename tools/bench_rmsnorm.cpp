// bench_rmsnorm.cpp -- steady-state bench for the rmsnorm row (Q32).
// Drives the shipped entry r3d_rad_rmsnorm at (rows M, length n, gain form)
// with random operands. Protocol laws inherited from bench_bf16_gemv:
//   * Liveness: random operands + nonzero count + FNV yhash per leg; the
//     R3D_DEBUG_RMS arm= line on stderr names the kernel each leg rode.
//   * Batch timing: reps batches of BATCH launches, median per-launch time --
//     rmsnorm kernels are short enough that one-launch timing is launch-tax.
//   * ROT rotates distinct x planes (each M*n*2 bytes) to defeat L2 when the
//     serving regime is a cold stream; rot=1 leaves rows L2-hot (decode).
// Same-binary A/B = the same build under R3D_RMS_VEC=0 and =1 in separate
// processes (the lever is a process-static read).
// Build/run (radiance-dev, gfx1100 pinned; keep -Wall on this file -- law):
//   hipcc -O1 -std=c++20 -I/opt/radiance/include/radiance -Wall \
//     /work/tools/bench_rmsnorm.cpp -o /scratch/brms -ldl
//   ROCR_VISIBLE_DEVICES=1 LIBR3D_SO=... R3D_RMS_VEC=<0|1> \
//     /scratch/brms <n> <M> <gain 0|1|2> <reps> [batch] [rot]
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
    const long n    = argc > 1 ? atol(argv[1]) : 4096;
    const long M    = argc > 2 ? atol(argv[2]) : 1024;
    const int  gain = argc > 3 ? atoi(argv[3]) : 1;
    const int  reps = argc > 4 ? atoi(argv[4]) : 30;
    const int  batch= argc > 5 ? atoi(argv[5]) : 20;
    const int  rot  = argc > 6 ? atoi(argv[6]) : 1;
    if (n <= 0 || M <= 0 || gain < 0 || gain > 2 || rot < 1 || rot > 4 || batch < 1) {
        printf("usage: bench_rmsnorm n M gain(0 none|1 bf16|2 f32) reps [batch] [rot<=4]\n");
        return 2; }

    void* dl = dlopen(getenv("LIBR3D_SO") ? getenv("LIBR3D_SO")
                                          : "/build/radiance_home/kernels/libr3d.so",
                      RTLD_NOW | RTLD_LOCAL);
    if (!dl) { printf("dlopen libr3d: %s\n", dlerror()); return 1; }
    auto entry = (int (*)(const RadArgs*, RadStream))dlsym(dl, "r3d_rad_rmsnorm");
    if (!entry) { printf("dlsym r3d_rad_rmsnorm: %s\n", dlerror()); return 1; }

    uint32_t st = 0x2468ace1u;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };
    auto rbf = [&]() { return (uint16_t)((rnd() & 0x8000u) | 0x3F80u | (rnd() & 0x7fu)); };

    const size_t plane = (size_t)M * n * 2;
    std::vector<void*> dx(rot);
    for (int p = 0; p < rot; ++p) {
        std::vector<uint16_t> h((size_t)M * n);
        for (auto& v : h) v = rbf();
        die("mallocx", hipMalloc(&dx[p], plane));
        die("copyx", hipMemcpy(dx[p], h.data(), plane, hipMemcpyHostToDevice));
    }
    void* dy; die("malty", hipMalloc(&dy, plane));
    void* dw = nullptr;
    if (gain) {
        const size_t wb = (size_t)n * (gain == 2 ? 4 : 2);
        die("mallocw", hipMalloc(&dw, wb));
        if (gain == 2) {   // f32 gains kept sane: 1 + u/2^16
            std::vector<float> f(n); for (auto& v : f) v = 1.0f + (float)(rnd() & 0xffff) / 65536.0f;
            die("copyw", hipMemcpy(dw, f.data(), wb, hipMemcpyHostToDevice));
        } else {
            std::vector<uint16_t> h(n); for (auto& v : h) v = rbf();
            die("copyw", hipMemcpy(dw, h.data(), wb, hipMemcpyHostToDevice));
        }
    }

    auto mk = [](void* d, uint32_t dt, int rank, std::vector<int64_t> sh,
                 std::vector<int64_t> stride) {
        RadTensor x{}; x.data = d; x.dtype = dt; x.rank = (uint32_t)rank;
        for (size_t i = 0; i < sh.size(); ++i) { x.shape[i] = sh[i]; x.stride[i] = stride[i]; }
        return x; };
    RadTensor* t = nullptr;
    die("mallocT", hipHostMalloc((void**)&t, sizeof(RadTensor) * 3));
    RadParam pv[] = { RAD_INT("n", n) };

    auto launch = [&](int p) {
        RadTensor b[3] = {
            mk(dx[p], RAD_BF16, 2, { M, n }, { n, 1 }),
            mk(dw, gain == 2 ? RAD_F32 : RAD_BF16, 1, { n }, { 1 }),
            mk(dy, RAD_BF16, 2, { M, n }, { n, 1 }),
        };
        memcpy(t, b, sizeof(b));
        RadArgs a{}; a.p = pv; a.n_p = 1; a.t = t; a.n_t = 3;
        return entry(&a, (RadStream)0);
    };

    if (launch(0) != 0) { printf("ENTRY REFUSED rc -- row not serving this shape\n"); return 3; }
    die("warm", hipDeviceSynchronize());

    std::vector<double> tms;
    for (int r = 0; r < reps; ++r) {
        timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int b = 0; b < batch; ++b) launch((r * batch + b) % rot);
        die("sync", hipDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC, &t1);
        tms.push_back(((t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec)) / batch);
    }
    std::sort(tms.begin(), tms.end());
    const double med = tms[tms.size() / 2];
    const double bytes = (double)plane * 2.0 + (gain ? (double)n * (gain == 2 ? 4 : 2) : 0.0);
    const double gbs = bytes / med / 1e9;

    std::vector<uint16_t> hy((size_t)M * n);
    die("copyy", hipMemcpy(hy.data(), dy, plane, hipMemcpyDeviceToHost));
    size_t nz = 0; unsigned long long yh = 1469598103934665603ull;
    for (const uint16_t v : hy) {
        nz += v != 0;
        yh = (yh ^ (v & 0xffu)) * 1099511628211ull;
        yh = (yh ^ (v >> 8))    * 1099511628211ull;
    }
    printf("ROW n=%ld M=%ld gain=%d med_ms=%.6f gbs=%.1f nz=%zu/%zu yhash=%016llx\n",
           n, M, gain, med * 1e3, gbs, nz, hy.size(), yh);
    return 0;
}
