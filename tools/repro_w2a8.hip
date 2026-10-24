// repro_w2a8.hip -- run gemm_nt_q_w2a8_decode on the card and libref's host gemm_nt_q
// row over the SAME operands, at every tile-boundary M. Two eras of purpose:
//   (1) it reproduced the stub-signature "cannot find symbol" launch failure that made
//       the row unlaunchable on any card (fixed 2026-10-24: the non-gfx11 host stub had
//       drifted from the real kernel's signature; HIP resolves launches by mangled name).
//   (2) it is now the row's numeric gate -- no kbench fixture holds w2a8 draft cases
//       (recording only recipes the target's op shapes present; the draft gemm band lives
//       in the drafter block), and --record-rows cannot describe the joined scale,zero
//       plane. The row's claim is bit-exactness to ref_gemm's gemm_zp; this measures it.
//
// Build/run (in radiance-dev on the card):
//   hipcc -O1 -std=c++20 -I/opt/radiance/include/radiance -w tools/repro_w2a8.hip \
//     -o /tmp/rw2 -ldl
//   LIBR3D_SO=$PWD/build_v111/radiance_home/kernels/libr3d.so \
//   LIBREF_SO=/opt/radiance/share/radiance/kernels/libref.so /tmp/rw2
// Exit 0 = every M launched and every output within tolerance (exact when bit-exact).
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "rad_abi.h"
#include "rad_plugin.h"
#include "rad_planes.h"

typedef const RadKernelInfo* (*AtFn)(int);
typedef int (*CountFn)();

static uint16_t rtne_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float bf16_f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f32_to_f16(float f) {   // round-to-nearest-even, exact for our small dyadic scales
    uint32_t u; memcpy(&u, &f, 4);
    int32_t e = (int32_t)((u >> 23) & 0xff) - 127 + 15;
    uint32_t s = (u >> 16) & 0x8000u;
    if (e <= 0) return (uint16_t)s;                       // our scales never underflow
    if (e >= 31) return (uint16_t)(s | 0x7c00u);
    uint32_t mant = u & 0x7fffffu;
    uint16_t h = (uint16_t)(s | ((uint32_t)e << 10) | (mant >> 13));
    if (mant & 0x1000u) ++h;                              // ties-away-ish: fine for dyadics
    return h;
}

int main() {
    void* dl = dlopen(getenv("LIBR3D_SO") ? getenv("LIBR3D_SO")
                                          : "/work/build_v111/radiance_home/kernels/libr3d.so",
                      RTLD_NOW | RTLD_LOCAL);
    if (!dl) { printf("dlopen libr3d: %s\n", dlerror()); return 1; }
    auto entry = (int (*)(const RadArgs*, RadStream))
                     dlsym(dl, "r3d_rad_gemm_nt_q_w2a8_decode");
    if (!entry) { printf("dlsym entry: %s\n", dlerror()); return 1; }
    void* hf = dlopen(getenv("LIBREF_SO") ? getenv("LIBREF_SO")
                                          : "/opt/radiance/share/radiance/kernels/libref.so",
                      RTLD_NOW | RTLD_LOCAL);
    if (!hf) { printf("dlopen libref: %s\n", dlerror()); return 1; }
    auto atf  = (AtFn)dlsym(hf, "rad_kernel_at");
    auto cntf = (CountFn)dlsym(hf, "rad_kernel_count");
    if (!atf || !cntf) { printf("libref kernel table missing\n"); return 1; }
    const RadKernelInfo* rk = nullptr;
    for (int k = 0; k < cntf(); ++k) {
        const RadKernelInfo* kk = atf(k);
        if (kk && !std::strcmp(kk->op, "gemm_nt_q") && kk->domain == RAD_DOMAIN_HOST) {
            printf("libref host row: %s\n", kk->name);
            if (!rk) rk = kk;
        }
    }
    if (!rk) { printf("libref has no host gemm_nt_q row\n"); return 1; }

    const long long N = 256, K = 1024, ngrp = K / 128, mpad = 16;
    const int Ms[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16 };

    uint32_t st = 0x2468ace1u;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };
    const bool TRIV = getenv("REPRO_TRIV") != nullptr;   // uniform A, z=0, s=1

    std::vector<int8_t>  hA(16 * K);
    for (auto& v : hA) v = TRIV ? (int8_t)1 : (int8_t)((rnd() % 8) - 4);
    std::vector<uint8_t> hB(N * (K / 4));
    for (auto& v : hB) v = (uint8_t)rnd();                       // four u2 codes per byte
    std::vector<float>   hAS(16);
    for (auto& v : hAS) v = TRIV ? 1.0f : 0.01f + 0.001f * (float)(rnd() % 16);
    // a_sum: the i8 row's per-group sums -- the oracle consumes it like any input, but a
    // truthful one keeps every product in bf16 range.
    std::vector<float>   hSUM(ngrp * mpad, 0.f);
    for (long long m = 0; m < 16; ++m)
        for (long long g = 0; g < ngrp; ++g) {
            float s = 0;
            for (long long k = g * 128; k < (g + 1) * 128; ++k) s += (float)hA[m * K + k];
            hSUM[g * mpad + m] = s;
        }

    // joined scale,zero blob: scale f16 ~0.003, zero u8 0..2 per (row, group)
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
                float s = 0.0030f + 0.000125f * (float)(rnd() % 32);   // dyadic: exact f16
                uint16_t h16 = TRIV ? (uint16_t)0x3c00 : f32_to_f16(s); // 1.0 or the dyadic scale
                memcpy(&scBlob[(size_t)(sc_off + (r * ngrp + g) * 2)], &h16, 2);
                scBlob[(size_t)(zp_off + r * ngrp + g)] = TRIV ? 0 : (uint8_t)(rnd() % 3);
            }
    }
    (void)rtne_bf16;

    // pinned host operands (host rows read host addresses; the device copy is separate)
    void *hA_p, *hB_p, *hAS_p, *hSUM_p, *hBlob_p, *hY_p, *hYref_p;
    hipHostMalloc(&hA_p,   hA.size());        memcpy(hA_p,   hA.data(),   hA.size());
    hipHostMalloc(&hB_p,   hB.size());        memcpy(hB_p,   hB.data(),   hB.size());
    hipHostMalloc(&hAS_p,  hAS.size() * 4);   memcpy(hAS_p,  hAS.data(),  hAS.size() * 4);
    hipHostMalloc(&hSUM_p, hSUM.size() * 4);  memcpy(hSUM_p, hSUM.data(), hSUM.size() * 4);
    hipHostMalloc(&hBlob_p, scBlob.size());   memcpy(hBlob_p, scBlob.data(), scBlob.size());
    hipHostMalloc(&hY_p,   16 * N * 2);
    hipHostMalloc(&hYref_p, 16 * N * 2);
    void *dA, *dB, *dAS, *dSUM, *dBlob, *dY;
    hipMalloc(&dA,   hA.size());        hipMemcpy(dA,   hA.data(),   hA.size(),   hipMemcpyHostToDevice);
    hipMalloc(&dB,   hB.size());        hipMemcpy(dB,   hB.data(),   hB.size(),   hipMemcpyHostToDevice);
    hipMalloc(&dAS,  hAS.size() * 4);   hipMemcpy(dAS,  hAS.data(),  hAS.size() * 4, hipMemcpyHostToDevice);
    hipMalloc(&dSUM, hSUM.size() * 4);  hipMemcpy(dSUM, hSUM.data(), hSUM.size() * 4, hipMemcpyHostToDevice);
    hipMalloc(&dBlob, scBlob.size());   hipMemcpy(dBlob, scBlob.data(), scBlob.size(), hipMemcpyHostToDevice);
    hipMalloc(&dY,   16 * N * 2);

    auto mk = [](void* d, uint32_t dt, int rank, std::vector<int64_t> sh,
                 std::vector<int64_t> stride) {
        RadTensor x{}; x.data = d; x.dtype = dt; x.rank = (uint32_t)rank;
        for (size_t i = 0; i < sh.size(); ++i) { x.shape[i] = sh[i]; x.stride[i] = stride[i]; }
        return x; };
    auto tensors = [&](void* dYp, void* a, void* b, void* as, void* sum, void* blob) {
        return std::vector<RadTensor>{
            mk(a, RAD_I8, 2, {16, K}, {K, 1}),
            mk(as, RAD_F32, 1, {16}, {1}),
            mk(b, RAD_U2, 2, {N, K}, {K, 1}),
            mk(blob, RAD_U8, 1, {(int64_t)blob ? (int64_t)0 : 0}, {1}),   // size set below
            mk(dYp, RAD_BF16, 2, {0, N}, {N, 1}),                         // shape[0]=M per case
            mk(sum, RAD_F32, 2, {ngrp, mpad}, {mpad, 1}),
        };
    };
    std::vector<RadTensor> host_ref = tensors(hYref_p, hA_p, hB_p, hAS_p, hSUM_p, hBlob_p);
    std::vector<RadTensor> host_dev = tensors(hY_p,    hA_p, hB_p, hAS_p, hSUM_p, hBlob_p);
    host_ref[3].shape[0] = blob; host_dev[3].shape[0] = blob;

    RadTensor* tdev = nullptr; RadTensor* tref = nullptr;
    hipHostMalloc((void**)&tdev, sizeof(RadTensor) * 6);
    hipHostMalloc((void**)&tref, sizeof(RadTensor) * 6);
    void* dptrs[6] = { dA, dAS, dB, dBlob, dY, dSUM };

    RadParam pv[] = { RAD_STR("dtype", "w2a8"), RAD_INT("N", N), RAD_INT("K", K),
                      RAD_INT("group", 128) };

    int fails = 0; bool exact_all = true;
    for (int M : Ms) {
        host_dev[4].shape[0] = M; host_ref[4].shape[0] = M;
        auto dv = host_dev; for (int i = 0; i < 6; ++i) dv[i].data = dptrs[i];
        memcpy(tdev, dv.data(), sizeof(RadTensor) * 6);
        memcpy(tref, host_ref.data(), sizeof(RadTensor) * 6);
        memset(hY_p, 0, (size_t)M * N * 2); memset(hYref_p, 0, (size_t)M * N * 2);

        RadArgs a{}; a.p = pv; a.n_p = 4; a.t = tdev; a.n_t = 6;
        const int rc = entry(&a, (RadStream)0);
        hipDeviceSynchronize();
        hipMemcpy(hY_p, dY, (size_t)M * N * 2, hipMemcpyDeviceToHost);
        std::vector<uint16_t> got((size_t)M * N);
        memcpy(got.data(), hY_p, got.size() * 2);

        RadArgs h{}; h.p = pv; h.n_p = 4; h.t = tref; h.n_t = 6;
        const int rrc = rk->launch(&h, (RadStream)nullptr);
        // libref host rows may thread internally: wait for the output to be byte-stable.
        std::vector<uint16_t> want((size_t)M * N);
        int idle = 0;
        for (int round = 0; round < 100 && idle < 5; ++round) {
            std::vector<uint16_t> snap((size_t)M * N);
            memcpy(snap.data(), hYref_p, snap.size() * 2);
            idle = (snap == want) ? idle + 1 : 0;
            want = snap;
            usleep(100 * 1000);
        }

        // Independent oracle: y[m,n] = a_scale[m] * SUM_g (is_g - z_g*asum_g) * s_g,
        // is_g = SUM_{k in g} a[k]*q[k] (exact int), ascending k then ascending g.
        auto oracle = [&](int m) {
            std::vector<float> row;
            for (long long n = 0; n < N; ++n) {
                double acc = 0;
                for (long long g = 0; g < ngrp; ++g) {
                    long long is = 0, asum = 0;
                    for (long long k = g * 128; k < (g + 1) * 128; ++k) {
                        int byte = hB[(size_t)(n * (K / 4)) + (k >> 2)];
                        int q = (byte >> (2 * (k & 3))) & 3;
                        // quant_act_i8's step order, the same law ref and the kernel read
                        long long ka = (k & ~12LL) | (((k >> 2) & 1) << 3) | (((k >> 3) & 1) << 2);
                        is += (long long)hA[(size_t)m * K + ka] * q;
                        asum += hA[(size_t)m * K + ka];
                    }
                    uint16_t h16; memcpy(&h16, &scBlob[(size_t)(sc_off + (n * ngrp + g) * 2)], 2);
                    uint32_t su = ((uint32_t)h16 & 0x8000u) << 16;
                    uint32_t em = h16 & 0x7fffu;
                    float sg;
                    if ((em & 0x7c00u) == 0) { su |= em << (16 - 1); memcpy(&sg, &su, 4); }
                    else { uint32_t f = (((em >> 10) + 112u) << 23) | ((em & 0x3ffu) << 13);
                           su |= f; memcpy(&sg, &su, 4); }
                    uint8_t z = scBlob[(size_t)(zp_off + n * ngrp + g)];
                    acc += ((double)is - (double)z * (double)asum) * (double)sg;
                }
                row.push_back((float)(hAS[m] * acc));
            }
            return row;
        };
        double se_o = 0, sw_o = 0, se_ok = 0, sw_ok = 0;
        for (int m = 0; m < M; ++m) {
            auto orom = oracle(m);
            for (long long n = 0; n < N; ++n) {
                double o = orom[n], w = bf16_f(want[(size_t)m * N + n]), g = bf16_f(got[(size_t)m * N + n]);
                se_o += (o - w) * (o - w); sw_o += w * w;
                se_ok += (o - g) * (o - g); sw_ok += g * g;
            }
        }
        if (getenv("REPRO_ORACLE"))
            printf("   oracle-vs-ref rel=%.3e  oracle-vs-kernel rel=%.3e\n",
                   sw_o > 0 ? std::sqrt(se_o / sw_o) : (se_o > 0 ? 1.0 : 0.0),
                   sw_ok > 0 ? std::sqrt(se_ok / sw_ok) : (se_ok > 0 ? 1.0 : 0.0));

        double se = 0, sw = 0; size_t diff = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            if (got[i] != want[i]) ++diff;
            double g = bf16_f(got[i]), w = bf16_f(want[i]);
            se += (g - w) * (g - w); sw += w * w;
        }
        const double rel = sw > 0 ? std::sqrt(se / sw) : (se > 0 ? 1.0 : 0.0);
        if (diff && getenv("REPRO_SAMPLE"))
            printf("   got[0..3]=%.4f %.4f %.4f %.4f want[0..3]=%.4f %.4f %.4f %.4f\n",
                   bf16_f(got[0]), bf16_f(got[1]), bf16_f(got[2]), bf16_f(got[3]),
                   bf16_f(want[0]), bf16_f(want[1]), bf16_f(want[2]), bf16_f(want[3]));
        const bool ok = rc == 0 && rrc == 0 && rel < 2e-2;
        if (diff) exact_all = false;
        printf("M=%-2d rc=%d ref_rc=%d rel_l2=%.3e exact=%s%s\n", M, rc, rrc, rel,
               diff ? "NO" : "YES", ok ? "" : "   <-- FAIL");
        if (!ok) ++fails;
    }
    printf("%s\n", fails ? "REPRO: failures"
                         : (exact_all ? "all M launch, BIT-EXACT to libref"
                                      : "all M launch, within tolerance (not bit-exact)"));
    return fails ? 2 : 0;
}
