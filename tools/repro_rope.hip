// repro_rope.hip -- standalone reproduction of the rope kbench failure.
// Builds the exact failing geometry, calls r3d_rad_rope through the plugin .so,
// compares element by element against the oracle's arithmetic (host, double-width).
// Build/run (in radiance-dev on the card): hipcc -O1 -std=c++20 -I/opt/radiance/include/radiance -w \
//   tools/repro_rope.hip -o /tmp/rr -ldl && LIBR3D_SO=$PWD/build/radiance_home/kernels/libr3d.so /tmp/rr 64
// UNIT=1 gives an impulse (x0=1, x1=0, pos from $POS): the outputs ARE cos/sin of the angle
// the kernel actually used. 0/65536 differing is the pass condition (one 1-ULP bf16 straggler
// at large M is sincosf-vs-double noise the kbench tolerance absorbs).
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rad_abi.h"

static uint16_t rtne_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float bf16_f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 2;
    const int HD = 128, NH = 8, NKV = 0;
    const double THETA = 1e7;
    const int SROW = (NH + 2 * NKV) * HD;

    uint16_t* hq = new uint16_t[(size_t)M * SROW];
    uint32_t st = 0x9e3779b9u;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };
    const bool unit = getenv("UNIT") != nullptr;
    if (unit) {
        memset(hq, 0, (size_t)M * SROW * 2);
        hq[0] = rtne_bf16(1.0f);                 // head 0, dim 0 = 1.0; dim 64 = 0.0
        for (int m = 0; m < M; ++m) hq[(size_t)m * SROW + 0] = rtne_bf16(1.0f);
    } else {
        for (int i = 0; i < M * SROW; ++i) {
            float v = ((rnd() & 0xffff) / 32768.0f) - 1.0f;   // [-1,1)
            hq[i] = rtne_bf16(v);
        }
    }
    int32_t* hp = new int32_t[M];
    for (int m = 0; m < M; ++m) hp[m] = unit ? (getenv("POS") ? atoi(getenv("POS")) : 1) : (int32_t)(rnd() % 60000);

    // expected: oracle arithmetic on host
    uint16_t* want = new uint16_t[(size_t)M * SROW];
    memcpy(want, hq, (size_t)M * SROW * 2);
    const int half = HD / 2;
    float inv[64];
    for (int i = 0; i < half; ++i)
        inv[i] = 1.0f / (float)std::pow(THETA, (double)((float)(2 * i) / (float)HD));
    for (int m = 0; m < M; ++m) {
        float p = (float)hp[m] / 1.0f;
        for (int h = 0; h < NH + NKV; ++h) {
            uint16_t* xr = want + (size_t)m * SROW + h * HD;
            for (int i = 0; i < half; ++i) {
                float ang = p * inv[i];
                float c = (float)std::cos((double)ang);
                float s = (float)std::sin((double)ang);
                float x0 = bf16_f(xr[i]), x1 = bf16_f(xr[i + half]);
                uint16_t a = rtne_bf16(x0 * c - x1 * s);
                uint16_t b = rtne_bf16(x0 * s + x1 * c);
                xr[i] = a; xr[i + half] = b;
            }
        }
    }

    // device copy + call through the plugin
    void* dl = dlopen(getenv("LIBR3D_SO") ? getenv("LIBR3D_SO") : "/build/radiance_home/kernels/libr3d.so", RTLD_NOW);
    if (!dl) { printf("dlopen: %s\n", dlerror()); return 1; }
    auto entry = (int (*)(const RadArgs*, RadStream))dlsym(dl, "r3d_rad_rope");
    if (!entry) { printf("dlsym: %s\n", dlerror()); return 1; }

    uint16_t* dq; int32_t* dp;
    hipMalloc(&dq, (size_t)M * SROW * 2);
    hipMalloc(&dp, (size_t)M * 4);
    hipMemcpy(dq, hq, (size_t)M * SROW * 2, hipMemcpyHostToDevice);
    hipMemcpy(dp, hp, (size_t)M * 4, hipMemcpyHostToDevice);

    RadTensor t_qkv{}, t_pos{};
    t_qkv.data = dq; t_qkv.dtype = RAD_BF16; t_qkv.rank = 2;
    t_qkv.shape[0] = M; t_qkv.shape[1] = SROW;
    t_qkv.stride[0] = SROW; t_qkv.stride[1] = 1;
    t_pos.data = dp; t_pos.dtype = RAD_I32; t_pos.rank = 1;
    t_pos.shape[0] = M; t_pos.stride[0] = 1;

    RadParam pv[] = {
        RAD_INT("M", M), RAD_INT("head_dim", HD), RAD_INT("n_head", NH),
        RAD_INT("n_head_kv", NKV), RAD_F64("theta", THETA), RAD_F64("scale", 1.0),
        RAD_STR("mode", "neox"),
    };
    RadTensor tensors[] = { t_qkv, t_pos };
    RadArgs a{};
    a.p = pv; a.n_p = sizeof(pv) / sizeof(pv[0]);
    a.t = tensors; a.n_t = 2;

    int rc = entry(&a, (RadStream)0);
    printf("entry rc=%d\n", rc);
    hipDeviceSynchronize();

    uint16_t* got = new uint16_t[(size_t)M * SROW];
    hipMemcpy(got, dq, (size_t)M * SROW * 2, hipMemcpyDeviceToHost);

    int bad = 0, shown = 0;
    for (int m = 0; m < M; ++m)
        for (int j = 0; j < SROW; ++j) {
            uint16_t w = want[(size_t)m * SROW + j], g = got[(size_t)m * SROW + j];
            if (w != g) {
                ++bad;
                if (shown++ < 12)
                    printf("m=%d el=%d (h=%d d=%d): want %.6f got %.6f (p=%d)\n",
                           m, j, j / HD, j % HD, bf16_f(w), bf16_f(g), hp[m]);
            }
        }
    printf("M=%d: %d/%d elements differ\n", M, bad, M * SROW);
    if (unit)
        printf("UNIT m=0: el0 got=%.6f (cos1=0.540302) el64 got=%.6f (sin1=0.841471)\n",
               bf16_f(got[0]), bf16_f(got[64]));
    return 0;
}
