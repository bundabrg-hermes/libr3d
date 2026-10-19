// repro_add.hip -- standalone reproduction of the add kbench failures (bf16 M>=2).
// Calls r3d_rad_add through the plugin .so under the shapes the fixture's failing cases
// use, element-compares against the oracle arithmetic (f32 sum, one bf16 narrowing).
// Build/run (in radiance-dev on the card): hipcc -O1 -std=c++20 -I/opt/radiance/include/radiance -w \
//   tools/repro_add.hip -o /tmp/ra -ldl && LIBR3D_SO=$PWD/build3/radiance_home/kernels/libr3d.so /tmp/ra
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rad_abi.h"

static uint16_t rtne_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float bf16_f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

typedef int (*entry_t)(const RadArgs*, RadStream);

static int run_case(entry_t entry, int M, int n, bool bcast_rank1, uint32_t dt, const char* name) {
    const size_t an = (size_t)M * n, bn = bcast_rank1 ? (size_t)n : (size_t)M * n;
    const size_t esz = dt == RAD_F32 ? 4 : 2;
    uint32_t st = 0x243f6a88u ^ (uint32_t)M;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };

    float* ha = new float[an]; float* hb = new float[bn];
    for (size_t i = 0; i < an; ++i) ha[i] = ((rnd() & 0xffff) / 32768.0f) - 1.0f;
    for (size_t i = 0; i < bn; ++i) hb[i] = ((rnd() & 0xffff) / 32768.0f) - 1.0f;

    // host encode + oracle expectation
    void* hA = malloc(an * esz); void* hB = malloc(bn * esz); void* hY = malloc(an * esz);
    memset(hY, 0, an * esz);
    for (size_t i = 0; i < an; ++i)
        if (dt == RAD_F32) ((float*)hA)[i] = ha[i]; else ((uint16_t*)hA)[i] = rtne_bf16(ha[i]);
    for (size_t i = 0; i < bn; ++i)
        if (dt == RAD_F32) ((float*)hB)[i] = hb[i]; else ((uint16_t*)hB)[i] = rtne_bf16(hb[i]);
    for (int m = 0; m < M; ++m)
        for (int i = 0; i < n; ++i) {
            const float u = dt == RAD_F32 ? ((float*)hA)[(size_t)m * n + i] : bf16_f(((uint16_t*)hA)[(size_t)m * n + i]);
            const float v = dt == RAD_F32 ? ((float*)hB)[(bcast_rank1 ? i : (size_t)m * n + i)]
                                          : bf16_f(((uint16_t*)hB)[bcast_rank1 ? i : (size_t)m * n + i]);
            const float s = u + v;
            if (dt == RAD_F32) ((float*)hY)[(size_t)m * n + i] = s;
            else ((uint16_t*)hY)[(size_t)m * n + i] = rtne_bf16(s);
        }

    void *dA, *dB, *dY;
    hipMalloc(&dA, an * esz); hipMalloc(&dB, bn * esz); hipMalloc(&dY, an * esz);
    hipMemcpy(dA, hA, an * esz, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, bn * esz, hipMemcpyHostToDevice);
    hipMemset(dY, 0, an * esz);

    RadTensor t_a{}, t_b{}, t_y{};
    t_a.data = dA; t_a.dtype = dt; t_a.rank = 2;
    t_a.shape[0] = M; t_a.shape[1] = n; t_a.stride[0] = n; t_a.stride[1] = 1;
    t_b.data = dB; t_b.dtype = dt;
    if (bcast_rank1) { t_b.rank = 1; t_b.shape[0] = n; t_b.stride[0] = 1; }
    else { t_b.rank = 2; t_b.shape[0] = M; t_b.shape[1] = n; t_b.stride[0] = n; t_b.stride[1] = 1; }
    t_y.data = dY; t_y.dtype = dt; t_y.rank = 2;
    t_y.shape[0] = M; t_y.shape[1] = n; t_y.stride[0] = n; t_y.stride[1] = 1;

    RadParam pv[] = { RAD_INT("M", M), RAD_INT("n", n),
                      RAD_STR("dtype", dt == RAD_F32 ? "f32" : "bf16") };
    RadTensor tensors[] = { t_a, t_b, t_y };
    RadArgs a{}; a.p = pv; a.n_p = 3; a.t = tensors; a.n_t = 3;

    const int rc = entry(&a, (RadStream)0);
    hipDeviceSynchronize();
    void* hGot = malloc(an * esz);
    hipMemcpy(hGot, dY, an * esz, hipMemcpyDeviceToHost);
    int bad = 0, shown = 0;
    for (size_t i = 0; i < an; ++i) {
        const bool diff = dt == RAD_F32 ? ((float*)hGot)[i] != ((float*)hY)[i]
                                        : ((uint16_t*)hGot)[i] != ((uint16_t*)hY)[i];
        if (diff && ++bad == 1 || diff && shown++ < 6)
            printf("  [%s] el %zu want %.6f got %.6f\n", name, i,
                   dt == RAD_F32 ? ((float*)hY)[i] : bf16_f(((uint16_t*)hY)[i]),
                   dt == RAD_F32 ? ((float*)hGot)[i] : bf16_f(((uint16_t*)hGot)[i]));
    }
    printf("%-40s rc=%d %d/%zu differ\n", name, rc, bad, an);
    free(ha); free(hb); free(hA); free(hB); free(hY); free(hGot);
    hipFree(dA); hipFree(dB); hipFree(dY);
    return bad;
}

int main() {
    void* dl = dlopen(getenv("LIBR3D_SO") ? getenv("LIBR3D_SO")
                                          : "/work/build3/radiance_home/kernels/libr3d.so", RTLD_NOW);
    if (!dl) { printf("dlopen: %s\n", dlerror()); return 1; }
    auto entry = (entry_t)dlsym(dl, "r3d_rad_add");
    if (!entry) { printf("dlsym: %s\n", dlerror()); return 1; }
    run_case(entry, 2, 5120, false, RAD_BF16, "bf16 M=2 full B [M,n]");
    run_case(entry, 2, 5120, true,  RAD_BF16, "bf16 M=2 bcast B [n] rank1");
    run_case(entry, 1, 5120, false, RAD_BF16, "bf16 M=1 full B [M,n]");
    run_case(entry, 2, 5120, false, RAD_F32,  "f32  M=2 full B [M,n]");
    run_case(entry, 33, 5120, true, RAD_BF16,  "bf16 M=33 bcast B [n] rank1");
    return 0;
}
