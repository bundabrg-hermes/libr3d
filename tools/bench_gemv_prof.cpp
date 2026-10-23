/* bench_m4_prof.cpp — steady-state profiler harness for the mxfp4 decode rows.
 * Pure host code (no __global__): dlopens libr3d.so, launches ONE row entry REPS
 * times at a chosen shape, prints device identity + wall time. Designed to be
 * wrapped by rocprofv3 (kernel-trace and --pmc counter passes).
 *
 * L2 policy: the serve regime has NO L2 reuse (whole 6.65 GiB weight set is
 * streamed per token, 96 MB L2 thrashes every plane before its next touch).
 * A replayed bench with a plane smaller than L2 measures a regime the serve
 * never runs in. Therefore: single plane with N*K/2 comfortably > 96 MB, and/or
 * PLANE_ROT rotating allocations. Rate is reported as weight-bytes/launch.
 *
 * usage: bench_m4_prof <which: tile|gemv> <M> <K> <N> <REPS> [PLANE_ROT]
 */
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include "radiance/rad_abi.h"
#include "radiance/rad_types.h"

typedef int (*LaunchFn)(const RadArgs*, RadStream);

static void die(const char* m, hipError_t e) { if (e == hipSuccess) return;
    fprintf(stderr, "%s: %s\n", m, hipGetErrorName(e)); exit(2);
}

int main(int argc, char** argv) {
    const char* which = argc > 1 ? argv[1] : "gemv";
    int M    = argc > 2 ? atoi(argv[2]) : 1;
    int K    = argc > 3 ? atoi(argv[3]) : 16384;
    int N    = argc > 4 ? atoi(argv[4]) : 16384;
    int reps = argc > 5 ? atoi(argv[5]) : 30;
    int rot  = argc > 6 ? atoi(argv[6]) : 1;   /* distinct B planes, L2 defeat */

    const char* so = getenv("LIBR3D_SO");
    void* h = dlopen(so ? so : "/build/radiance_home/kernels/libr3d.so", RTLD_NOW);
    if (!h) { printf("dlopen: %s\n", dlerror()); return 1; }
    const int is_fp8  = strncmp(which, "fp8", 3) == 0;
    LaunchFn fn = (LaunchFn)dlsym(h,
                    strcmp(which, "tile") == 0  ? "r3d_rad_gemm_nt_q" :
                    strcmp(which, "fp8tile") == 0 ? "r3d_rad_gemm_nt_q_fp8" :
                    is_fp8 ? "r3d_rad_gemm_nt_q_fp8_decode" : "r3d_rad_gemm_nt_q_mxfp4_decode");
    if (!fn) { printf("dlsym fail\n"); return 1; }

    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) die("getprops", hipGetLastError());
    printf("DEV name=%s arch=%s l2Bytes=%zu memGB=%.1f\n", prop.name, prop.gcnArchName,
           prop.l2CacheSize, prop.totalGlobalMem / 1073741824.0);

    int as_cols = (K + 127) / 128, bs_cols = (K + 31) / 32;
    if (is_fp8) bs_cols = (K + 127) / 128;              /* fp8a8: 128-block b grid */
    const size_t bbytes = (size_t)N * (is_fp8 ? K : K / 2);
    const size_t bsbytes = (size_t)N * bs_cols * (is_fp8 ? 2 : 1);   /* bf16 vs u8 */
    size_t abytes = (size_t)M * K;
    uint8_t *dA, *dBS; float *dAS; uint16_t *dY;
    die("mallocA", hipMalloc((void**)&dA, abytes));
    die("mallocBS", hipMalloc((void**)&dBS, bsbytes));
    die("mallocAS", hipMalloc((void**)&dAS, (size_t)M * as_cols * 4));
    die("mallocY", hipMalloc((void**)&dY, (size_t)M * N * 2));
    uint8_t** dB = new uint8_t*[rot];
    for (int r = 0; r < rot; r++) die("mallocB", hipMalloc((void**)&dB[r], bbytes));

    {   /* fill once host-side, random, upload */
        uint8_t* hA = new uint8_t[abytes];  uint8_t* hB = new uint8_t[bbytes];
        uint8_t* hBS = new uint8_t[bsbytes]; float* hAS = new float[(size_t)M * as_cols];
        srand(777);
        for (size_t i = 0; i < abytes; i++) hA[i] = (uint8_t)(rand() % 224) & 0x7e;
        for (size_t i = 0; i < bbytes; i++) hB[i] = (uint8_t)rand();
        for (size_t i = 0; i < bsbytes; i++) hBS[i] = (uint8_t)(115 + rand() % 25);
        if (is_fp8) {   /* bf16 scale plane: high 16 bits of a small float, u8-pattern exponents */
            uint16_t* w = (uint16_t*)hBS;
            for (size_t i = 0; i < bsbytes / 2; i++) {
                float f = 0.001f + 0.01f * (rand() % 1000) / 1000.f;
                uint32_t b; memcpy(&b, &f, 4);
                w[i] = (uint16_t)(b >> 16);
            }
        }
        for (size_t i = 0; i < (size_t)M * as_cols; i++) hAS[i] = 0.001f + 0.01f * (rand() % 1000) / 1000.f;
        die("cpyA", hipMemcpy(dA, hA, abytes, hipMemcpyHostToDevice));
        for (int r = 0; r < rot; r++) die("cpyB", hipMemcpy(dB[r], hB, bbytes, hipMemcpyHostToDevice));
        die("cpyBS", hipMemcpy(dBS, hBS, bsbytes, hipMemcpyHostToDevice));
        die("cpyAS", hipMemcpy(dAS, hAS, (size_t)M * as_cols * 4, hipMemcpyHostToDevice));
    }

    RadTensor t[7] = {};
    RadTensor* T = t;
    T[0] = (RadTensor){dA, RAD_F8E4M3, 2, {M, K}, {K, 1}};
    T[1] = (RadTensor){dAS, RAD_F32, 2, {M, as_cols}, {as_cols, 1}};
    T[2] = (RadTensor){dB[0], is_fp8 ? RAD_F8E4M3 : RAD_FP4E2M1, 2, {N, K}, {K, 1}};
    T[3] = (RadTensor){dBS, is_fp8 ? RAD_BF16 : RAD_E8M0, 2, {N, bs_cols}, {bs_cols, 1}};
    T[4] = (RadTensor){dY, RAD_BF16, 2, {M, N}, {N, 1}};
    T[5] = (RadTensor){nullptr, RAD_F32, 0, {0}, {0}};
    T[6] = (RadTensor){nullptr, RAD_BF16, 0, {0}, {0}};
    RadParam p[] = {
        RAD_STR("dtype", is_fp8 ? "fp8a8" : "mxfp4a8"), RAD_INT("N", N), RAD_INT("K", K),
        RAD_INT("group", is_fp8 ? 128 : 32),
    };
    RadArgs args = { t, 7, p, (int)(sizeof p / sizeof p[0]), nullptr, 0, nullptr, 0, 0 };

    /* entry once (also the correctness proof that rc==0), then reps with rotation */
    int rc = fn(&args, 0);
    if (rc != 0) { printf("ENTRY rc=%d (nonzero: unsupported for this shape)\n", rc); return 3; }
    die("sync0", hipDeviceSynchronize());
    timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < reps; i++) {
        T[2].data = dB[i % rot];
        int r2 = fn(&args, 0);
        if (r2 != 0) { printf("launch %d rc=%d\n", i, r2); return 3; }
    }
    die("syncEnd", hipDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    double wb = (double)bbytes;
    printf("RESULT which=%s M=%d K=%d N=%d rot=%d reps=%d total_ms=%.2f per_ms=%.3f "
           "GBps=%.1f (plane=%.1fMB)\n",
           which, M, K, N, rot, reps, ms, ms / reps,
           wb * reps / (ms / 1e3) / 1e9, bbytes / 1048576.0);
    return 0;
}
