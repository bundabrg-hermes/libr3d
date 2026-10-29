// q62_probe: minimal kv_store_fp8 + fused-fp8 probe against the ABI encoder oracle.
// M=2, nkv=1, hd=8? no -- hd must be %32/divisible: use hd=64, nq=0 (k-only fused),
// bs=4, nb=2. k plane constants with known fp8 answers. Print cache bytes.
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include "rad_abi.h"
#include "rad_plugin.h"

static uint16_t bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }
static float bfr(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char** argv) {
    void* dl = dlopen(argv[1], RTLD_NOW);
    if (!dl) { printf("dlopen %s\n", dlerror()); return 1; }
    auto get = [&](const char* n) { return (int (*)(const RadArgs*, RadStream))dlsym(dl, n); };
    int (*ks8)(const RadArgs*, RadStream) = get("r3d_rad_kv_store_fp8");
    if (!ks8) { printf("dlsym kv_store_fp8 failed\n"); return 1; }

    const int HD = 64, NKV = 1, BS = 4, NB = 2, M = 2;
    // k plane: token0: 1.0, 448, 449, -448, 0.001, tiny, 2.5, 300 then rest 1.0
    std::vector<uint16_t> hk(M * NKV * HD, bf(1.0f)), hv(M * NKV * HD, bf(2.0f));
    hk[0]=bf(1.0f); hk[1]=bf(448.0f); hk[2]=bf(449.0f); hk[3]=bf(-448.0f);
    hk[4]=bf(0.001f); hk[5]=bf(1.0f/512/2); hk[6]=bf(2.5f); hk[7]=bf(300.0f);
    std::vector<int32_t> slot{0, 5};
    std::vector<uint8_t> cache((size_t)NB*NKV*BS*2*HD, 0xA5);

    uint16_t *dk, *dv; int32_t* ds; uint8_t* dc;
    hipMalloc(&dk, hk.size()*2); hipMemcpy(dk, hk.data(), hk.size()*2, hipMemcpyHostToDevice);
    hipMalloc(&dv, hv.size()*2); hipMemcpy(dv, hv.data(), hv.size()*2, hipMemcpyHostToDevice);
    hipMalloc(&ds, slot.size()*4); hipMemcpy(ds, slot.data(), slot.size()*4, hipMemcpyHostToDevice);
    hipMalloc(&dc, cache.size()); hipMemset(dc, 0xA5, cache.size());

    RadTensor t[4]{};
    t[0].data=dk; t[0].dtype=RAD_BF16;   t[0].rank=2; t[0].shape[0]=M; t[0].shape[1]=NKV*HD;
    t[0].stride[0]=NKV*HD; t[0].stride[1]=1;
    t[1]=t[0]; t[1].data=dv;
    t[2].data=ds; t[2].dtype=RAD_I32; t[2].rank=1; t[2].shape[0]=M; t[2].stride[0]=1;
    t[3].data=dc; t[3].dtype=RAD_F8E4M3; t[3].rank=4;
    t[3].shape[0]=NB; t[3].shape[1]=NKV; t[3].shape[2]=BS; t[3].shape[3]=2*HD;
    t[3].stride[0]=(int64_t)NKV*BS*2*HD; t[3].stride[1]=(int64_t)BS*2*HD; t[3].stride[2]=2*HD; t[3].stride[3]=1;

    RadParam pv[] = { RAD_INT("head_dim", HD), RAD_INT("block_size", BS),
                      RAD_INT("n_head_kv", NKV), RAD_STR("kv_dtype", "fp8_e4m3"),
                      { "k_scale", RAD_P_F64, 0, 0, nullptr, 1.0 },
                      { "v_scale", RAD_P_F64, 0, 0, nullptr, 1.0 } };
    RadArgs a{}; a.t=t; a.n_t=4; a.p=pv; a.n_p=6;
    int rc = ks8(&a, (RadStream)0);
    hipDeviceSynchronize();
    hipMemcpy(cache.data(), dc, cache.size(), hipMemcpyDeviceToHost);
    printf("rc=%d\n", rc);

    int bad = 0;
    for (int tk = 0; tk < M; ++tk) {
        int sl = slot[tk];
        int b = sl/BS, off = sl%BS;
        for (int d = 0; d < HD; ++d) {
            uint8_t ek = rad_f32_to_fp8e4m3(bfr(hk[tk*NKV*HD+d]) * 1.0f);
            uint8_t ev = rad_f32_to_fp8e4m3(bfr(hv[tk*NKV*HD+d]) * 1.0f);
            uint8_t gk = cache[((b*NKV+0)*BS+off)*2*HD + d];
            uint8_t gv = cache[((b*NKV+0)*BS+off)*2*HD + HD + d];
            if (gk != ek || gv != ev) {
                if (bad < 12) printf("t=%d d=%d k: got %02x want %02x (k=%.6g) | v: got %02x want %02x\n",
                                     tk, d, gk, ek, bfr(hk[tk*NKV*HD+d]), gv, ev);
                ++bad;
            }
        }
    }
    // untouched slots must stay poisoned
    int red = 0;
    for (int s = 0; s < NB*BS; ++s) {
        if (s==0 || s==5) continue;
        for (int e = 0; e < 2*HD; ++e)
            if (cache[((s/BS)*NKV+0)*BS*2*HD + (s%BS)*2*HD + e] != 0xA5) { ++red; break; }
    }
    printf("bad=%d redzones_walked=%d\n", bad, red);
    return (rc || bad || red) ? 2 : 0;
}
