// bench_m64.cpp -- steady-state bench for the bf16 m64 WMMA prefill tile (Q1).
// Times r3d_gemm_nt_m64_launch directly (dlopen; the symbol is C++-mangled -- the
// dlsym string below must match `nm -D libr3d.so | grep m64_launch`).
//
// Shape law (Q1 queue entry): the MT>=2 staging gate means one-row-tile and
// multi-row-tile regimes are DIFFERENT ARMS, so the runner always benches both
// (M=64 is MT=1 and must show parity/negligence, M>=128 is the gated-on regime).
// Rotation only engages at K % 4096 == 0; odd-K rows (K=3584, 4104) are the
// control that the gate holds.
//
// L2 law (adopted card protocol): ROT rotates distinct B planes so a repeat does
// not measure an L2-resident regime. bf16 B plane = N*K*2 bytes; keep ROT*N*K*2
// past 96 MB where possible (ROT=2 minimum at N=11008). A is the activation
// plane (small at these M) and stays shared.
//
// Bit-identity law: the vec arm is a staging-only change; every arm must produce
// a byte-identical Y for identical operands. With argv[7] set, dump Y; the
// runner cmp -s's the dumps across arms before trusting a single timing column.
//
// Liveness law: random bf16 operands, nonzero-output count printed with rate.
//
// Build/run (in radiance-dev ON the card, gfx1100 pinned via ROCR_VISIBLE_DEVICES):
//   hipcc -O1 -std=c++20 -I/opt/radiance/include/radiance -w tools/bench_m64.cpp \
//     -o /tmp/bm64 -ldl
//   ROCR_VISIBLE_DEVICES=1 LIBR3D_SO=$PWD/build_v128/radiance_home/kernels/libr3d.so \
//     R3D_M64_VEC=1 R3D_DEBUG_GEMM=1 /tmp/bm64 <M> <N> <K> <REPS> [ROT] [dump.bin]
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <vector>

static void die(const char* m, hipError_t e) { if (e == hipSuccess) return;
    fprintf(stderr, "%s: %s\n", m, hipGetErrorName(e)); exit(2); }

typedef int (*m64_launch_t)(const void*, const void*, void*, float*,
                            int, int, int, int64_t, int64_t, int64_t, hipStream_t);

int main(int argc, char** argv) {
    const int      M    = argc > 1 ? atoi(argv[1]) : 1024;
    const int      N    = argc > 2 ? atoi(argv[2]) : 4096;
    const int      K    = argc > 3 ? atoi(argv[3]) : 4096;
    const int      reps = argc > 4 ? atoi(argv[4]) : 30;
    const int      rot  = argc > 5 ? atoi(argv[5]) : 2;
    const char*    dump = argc > 6 ? argv[6] : nullptr;
    if (M < 1 || N < 1 || K < 16 || reps < 3 || rot < 1 || rot > 4) {
        printf("usage: bench_m64 M N K REPS ROT(<=4) [dump.bin]\n"); return 2; }

    void* dl = dlopen(getenv("LIBR3D_SO") ? getenv("LIBR3D_SO")
                                          : "/build/radiance_home/kernels/libr3d.so",
                      RTLD_NOW | RTLD_LOCAL);
    if (!dl) { printf("dlopen libr3d: %s\n", dlerror()); return 1; }
    auto launch = (m64_launch_t)dlsym(dl, "_Z22r3d_gemm_nt_m64_launchPKvS0_PvPfiiilllP12ihipStream_t");
    if (!launch) { printf("dlsym m64_launch: %s\n", dlerror()); return 1; }

    hipDeviceProp_t prop;
    die("getprops", hipGetDeviceProperties(&prop, 0));
    printf("DEV name=%s arch=%s l2Bytes=%zu\n", prop.name, prop.gcnArchName, prop.l2CacheSize);

    // Which arm DO WE EXPECT for this shape? Mirrors the launcher's gates so a
    // bench line can never quietly measure a different arm than it claims.
    const int MT = (M + 63) / 64;
    const bool vec   = getenv("R3D_M64_VEC") != nullptr;
    const bool ronen = getenv("R3D_M64_ROT") != nullptr;
    const bool vec_ok = vec && MT >= 2 && (K % 8 == 0);   // pitch=K here; bases 256B
    const bool expect_rot = vec_ok && ronen && (K % 4096 == 0);
    printf("ARM expected=%s MT=%d K%%4096=%d\n",
           expect_rot ? "vec+rot" : vec_ok ? "vec" : "scalar", MT, K % 4096 == 0);

    uint32_t st = 0x9e3779b9u;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };
    // plausible bf16: sign + mantissa random, exponent in [118,138) -- no zeros,
    // no denormals, so liveness and accumulation are honest.
    auto bf16 = [&]() { uint32_t r = rnd();
        return (uint16_t)(((r & 0x8000u)) | ((118u + (r % 20u)) << 7) | ((r >> 9) & 0x7fu)); };

    std::vector<uint16_t> hA((size_t)M * K);
    for (auto& v : hA) v = bf16();
    std::vector<uint16_t> hB((size_t)N * K);
    for (auto& v : hB) v = bf16();

    void *dA, *dY; std::vector<void*> dB(rot);
    die("mallocA", hipMalloc(&dA, hA.size() * 2));
    die("copyA",   hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
    die("mallocY", hipMalloc(&dY, (size_t)M * N * 2));
    for (int p = 0; p < rot; ++p) {
        die("mallocB", hipMalloc(&dB[p], hB.size() * 2));
        die("copyB",   hipMemcpy(dB[p], hB.data(), hB.size() * 2, hipMemcpyHostToDevice));
        if (p) { // rotate a second distinct fill into plane 1.. so planes differ
            std::vector<uint16_t> alt(hB.size());
            for (auto& v : alt) v = bf16();
            die("copyBalt", hipMemcpy(dB[p], alt.data(), alt.size() * 2, hipMemcpyHostToDevice));
        }
    }

    auto call = [&](int plane) {
        return launch((const void*)dA, (const void*)dB[plane % rot], dY, nullptr,
                      M, N, K, K, K, N, (hipStream_t)nullptr);
    };
    if (call(0) != 0) { printf("LAUNCH-REJECTED (m64 refused the shape)\n"); return 3; }
    die("sync0", hipDeviceSynchronize());

    // warmup (3 launches, also the R3D_DEBUG_GEMM once-line lands before timing)
    for (int i = 0; i < 3; ++i) call(i % rot);
    die("syncw", hipDeviceSynchronize());

    std::vector<float> ms(reps);
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    for (int r = 0; r < reps; ++r) {
        die("ev0", hipEventRecord(e0));
        call(r % rot);
        die("ev1", hipEventRecord(e1));
        die("sync", hipEventSynchronize(e1));
        float t; die("el", hipEventElapsedTime(&t, e0, e1));
        ms[r] = t;
    }
    std::sort(ms.begin(), ms.end());
    const double med = ms[reps / 2];
    const double tflops = 2.0 * M * (double)N * K / (med * 1e-3) / 1e12;
    printf("TIME med_ms=%.3f min_ms=%.3f p75_ms=%.3f tflops=%.1f reps=%d rot=%d\n",
           med, ms.front(), ms[(reps * 3) / 4], tflops, reps, rot);

    // liveness
    std::vector<uint16_t> hY((size_t)M * N);
    die("copyY", hipMemcpy(hY.data(), dY, hY.size() * 2, hipMemcpyDeviceToHost));
    size_t nz = 0; for (auto v : hY) nz += (v != 0);
    printf("LIVE yElems=%zu nonzero=%zu (%.1f%%)\n", hY.size(), nz, 100.0 * nz / hY.size());
    if (nz == 0) { printf("DEAD-ROW: outputs all zero, timing is a lie\n"); return 4; }
    if (dump) {
        FILE* f = fopen(dump, "wb");
        if (!f) { printf("dump open failed\n"); return 5; }
        fwrite(hY.data(), 2, hY.size(), f); fclose(f);
        printf("DUMP %s\n", dump);
    }
    return 0;
}
