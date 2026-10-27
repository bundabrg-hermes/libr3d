// q49_probe.cpp -- Q49 pre-integration probe for gated_quant_fp8_vec + mul_kernel_vec.
// Matrix: serve shape, group widths (cpg 2..32), both act forms, ob present, pitched
// (%8 and %8!=0), base alignment (+8 accepted-reject / +16 accept), ragged n%g, non-
// pow2 g, cpg>32, f32 dtype, mul boundary n=2048, small n, ragged n%8, bcast.
// Laws: cross-arm FNV must be ZERO-diff (bit-identity class); red-zone sentinels;
// R3D_DEBUG_VEC arm lines ordered with LEG lines (wrapper checks want vs got).
// Run: ROCR_VISIBLE_DEVICES=1 LIBR3D_SO=... R3D_VEC_GATE=<0|1> R3D_VEC_MUL=<0|1> \
//      R3D_DEBUG_VEC=1 q49p 2>&1 | tee log
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cstdio>
#include <vector>
#include "rad_abi.h"
#include "rad_types.h"

static void die(const char* m, hipError_t e) { if (e == hipSuccess) return;
    fprintf(stderr, "FATAL %s: %s\n", m, hipGetErrorName(e)); exit(2); }
static uint32_t st = 0xB00B1E5u;
static uint32_t rnd() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; }
static uint16_t rbf() {
    const uint32_t r = rnd();
    if (!getenv("Q4P_NONAN") && (r & 0xffffu) < 64u) return (uint16_t)((r & 0x8000u) | 0x7FC0u);
    return (uint16_t)(((r & 0x8000u) << 1) | ((r >> 17) & 0x8000u) | (r & 0x7FFFu)); }
static unsigned long long fnv(unsigned long long h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h; }
static RadTensor mk(void* d, uint32_t dt, int rank, std::vector<int64_t> sh,
                    std::vector<int64_t> stride) {
    RadTensor x{}; x.data = d; x.dtype = dt; x.rank = (uint32_t)rank;
    for (size_t i = 0; i < sh.size(); ++i) { x.shape[i] = sh[i]; x.stride[i] = stride[i]; }
    return x; }

struct Leg { const char* id; int mul; int64_t n, g, rows; int ob; const char* act;
             int64_t extra_pitch;   // +elements on gu/A row pitch
             int64_t base_off;      // +bytes on data bases (alignment leg)
             int f32gu; int bcast;
             const char* want; };

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    void* dl = dlopen(getenv("LIBR3D_SO") ? getenv("LIBR3D_SO") : "libr3d.so", RTLD_NOW | RTLD_LOCAL);
    if (!dl) { printf("dlopen: %s\n", dlerror()); return 1; }
    typedef int (*fn_t)(const RadArgs*, RadStream);
    auto gate_e = (fn_t)dlsym(dl, "r3d_rad_gated_quant_fp8");
    auto mul_e  = (fn_t)dlsym(dl, "r3d_rad_mul");
    if (!gate_e || !mul_e) { printf("dlsym\n"); return 1; }

    Leg legs[] = {
        { "g01-serve",   0, 12288, 128, 1, 0, "silu",    0, 0, 0, 0, "vec" },
        { "g02-verify",  0, 12288, 128, 5, 0, "silu",    0, 0, 0, 0, "vec" },
        { "g03-ob",      0, 12288, 128, 1, 1, "silu",    0, 0, 0, 0, "vec" },
        { "g04-g64",     0, 12288,  64, 1, 1, "silu",    0, 0, 0, 0, "vec" },
        { "g05-g32",     0, 12288,  32, 2, 0, "silu",    0, 0, 0, 0, "vec" },
        { "g06-g16",     0, 12288,  16, 1, 0, "silu",    0, 0, 0, 0, "vec" },
        { "g07-g256",    0, 12288, 256, 1, 0, "silu",    0, 0, 0, 0, "vec" },
        { "g08-sigmoid", 0, 12288, 128, 1, 0, "sigmoid", 0, 0, 0, 0, "vec" },
        { "g09-n4096",   0,  4096, 128, 3, 0, "silu",    0, 0, 0, 0, "vec" },
        { "g10-pitch8",  0, 12288, 128, 1, 0, "silu",    8, 0, 0, 0, "vec" },
        { "g11-pitch3",  0, 12288, 128, 1, 0, "silu",    3, 0, 0, 0, "scalar" },
        { "g12-base8",   0, 12288, 128, 1, 0, "silu",    0, 8, 0, 0, "scalar" },
        { "g13-base16",  0, 12288, 128, 1, 0, "silu",    0, 16, 0, 0, "vec" },
        { "g14-raggedn", 0, 12000, 128, 1, 0, "silu",    0, 0, 0, 0, "scalar" },
        { "g15-g100",    0, 12288, 100, 1, 0, "silu",    0, 0, 0, 0, "scalar" },
        { "g16-g512",    0, 12288, 512, 1, 0, "silu",    0, 0, 0, 0, "scalar" },
        { "g17-f32gu",   0, 12288, 128, 1, 0, "silu",    0, 0, 1, 0, "scalar" },
        { "m01-serve",   1,  4096,   0, 1, 0, nullptr,   0, 0, 0, 0, "vec" },
        { "m02-verify",  1,  4096,   0, 5, 0, nullptr,   0, 0, 0, 0, "vec" },
        { "m03-bound",   1,  2048,   0, 1, 0, nullptr,   0, 0, 0, 0, "vec" },
        { "m04-small",   1,  1024,   0, 1, 0, nullptr,   0, 0, 0, 0, "scalar" },
        { "m05-ragged",  1,  4095,   0, 1, 0, nullptr,   0, 0, 0, 0, "scalar" },
        { "m06-bcast",   1,  4096,   0, 5, 0, nullptr,   0, 0, 0, 1, "scalar" },
        { "m07-pitch8",  1,  4096,   0, 1, 0, nullptr,   8, 0, 0, 0, "vec" },
        { "m08-base8",   1,  4096,   0, 1, 0, nullptr,   0, 8, 0, 0, "scalar" },
        { "m09-head",    1,   256,   0, 1, 0, nullptr,   0, 0, 0, 0, "scalar" },
    };
    RadTensor* t = nullptr; die("hostT", hipHostMalloc((void**)&t, sizeof(RadTensor) * 4));

    for (const Leg& L : legs) {
        // ---------- planes + red zones ----------
        const int64_t gu_n = L.mul ? L.n : 2 * L.n;                    // row elements (logical)
        const int64_t gu_p = gu_n + L.extra_pitch;                     // row pitch
        const size_t RZ = 64;
        std::vector<void*> raws; std::vector<size_t> szs;
        const int64_t ngrp = L.g ? L.n / L.g + (L.n % L.g ? 1 : 0) : 1;
        std::vector<uint8_t> h;
        // gate/up (or A/B) planes
        { h.assign((size_t)L.rows * gu_p * (L.f32gu && !L.mul ? 4 : 2), 0);
          auto* p16 = (uint16_t*)h.data();
          for (size_t i = 0; i < h.size() / 2; ++i) p16[i] = rbf();
          if (L.f32gu && !L.mul) { auto* p32 = (float*)h.data();
              for (size_t i = 0; i < h.size() / 4; ++i) p32[i] = (float)(int16_t)((uint16_t*)h.data())[i] * 0.01f; } }
        void *rgu, *rq, *rsc, *rob, *ra, *rb, *ry;
        auto alloc_rz = [&](size_t bytes, void** out) {
            void* raw; die("alloc", hipMalloc(&raw, RZ + 16 + bytes + RZ));
            std::vector<uint8_t> z(RZ, 0xCDu);
            die("rzpre", hipMemcpy(raw, z.data(), RZ, hipMemcpyHostToDevice));
            die("rzpost", hipMemcpy((char*)raw + RZ + 16 + bytes, z.data(), RZ, hipMemcpyHostToDevice));
            raws.push_back(raw); szs.push_back(bytes);
            *out = (char*)raw + RZ; };
        alloc_rz(h.size(), &rgu); die("cgu", hipMemcpy(rgu, h.data(), h.size(), hipMemcpyHostToDevice));
        alloc_rz((size_t)L.rows * L.n, &rq);
        alloc_rz((size_t)L.rows * ngrp * 4, &rsc);
        alloc_rz((size_t)L.rows * L.n * 2, &rob);
        alloc_rz(h.size(), &ra);  die("ca",  hipMemcpy(ra, h.data(), h.size(), hipMemcpyHostToDevice));
        std::vector<uint8_t> hb(h.size()); auto* q16 = (uint16_t*)hb.data();
        for (size_t i = 0; i < hb.size() / 2; ++i) q16[i] = rbf();
        alloc_rz(hb.size(), &rb); die("cb", hipMemcpy(rb, hb.data(), hb.size(), hipMemcpyHostToDevice));
        alloc_rz((size_t)L.rows * L.n * 2, &ry);
        char* bgu = (char*)rgu + L.base_off; char* ba = (char*)ra + L.base_off;
        char* bb = (char*)rb + L.base_off;   char* by = (char*)ry + L.base_off;
        char* bq = (char*)rq + L.base_off;   char* bs = (char*)rsc + L.base_off;
        char* bo = (char*)rob + L.base_off;

        int rc; unsigned long long hsh = 1469598103934665603ull;
        if (!L.mul) {
            RadParam pv[] = { RAD_INT("n", L.n), RAD_INT("group", L.g), RAD_STR("act", L.act) };
            RadTensor b4[4];
            b4[0] = mk(bgu, L.f32gu ? RAD_F32 : RAD_BF16, 2, { L.rows, gu_n }, { gu_p, 1 });
            b4[1] = mk(bq, RAD_F8E4M3, 2, { L.rows, L.n }, { L.n, 1 });
            b4[2] = mk(bs, RAD_F32, 2, { L.rows, ngrp }, { ngrp, 1 });
            b4[3] = L.ob ? mk(bo, RAD_BF16, 2, { L.rows, L.n }, { L.n, 1 }) : mk(nullptr, RAD_BF16, 2, { 0, 0 }, { 0, 1 });
            memcpy(t, b4, sizeof(b4));
            RadArgs a{}; a.p = pv; a.n_p = 3; a.t = t; a.n_t = L.ob ? 4 : 3;
            rc = gate_e(&a, (RadStream)0);
            std::vector<uint8_t> oq((size_t)L.rows * L.n), osc((size_t)L.rows * ngrp * 4);
            std::vector<uint16_t> oo((size_t)L.rows * L.n);
            if (rc == 0) {
                die("cq", hipMemcpy(oq.data(), rq, oq.size(), hipMemcpyDeviceToHost));
                die("cs", hipMemcpy(osc.data(), rsc, osc.size(), hipMemcpyDeviceToHost));
                hsh = fnv(hsh, oq.data(), oq.size()); hsh = fnv(hsh, osc.data(), osc.size());
                if (L.ob) { die("co", hipMemcpy(oo.data(), rob, oo.size() * 2, hipMemcpyDeviceToHost));
                            hsh = fnv(hsh, oo.data(), oo.size() * 2); }
                if (getenv("Q4P_DUMP")) {
                    std::string f = std::getenv("Q4P_DUMP"); f += "."; f += L.id;
                    FILE* fp = fopen(f.c_str(), "wb");
                    fwrite(oq.data(), 1, oq.size(), fp); fwrite(osc.data(), 1, osc.size(), fp);
                    if (L.ob) fwrite(oo.data(), 2, oo.size(), fp);
                    fclose(fp);
                    std::string fi = f + ".in";
                    FILE* fi2 = fopen(fi.c_str(), "wb");
                    fwrite(h.data(), 1, h.size(), fi2); fclose(fi2); }
            }
        } else {
            RadParam pv[] = { RAD_INT("n", L.n), RAD_STR("dtype", "bf16") };
            RadTensor b3[3];
            b3[0] = mk(ba, RAD_BF16, 2, { L.rows, L.n }, { gu_p, 1 });
            if (L.bcast) b3[1] = mk(bb, RAD_BF16, 1, { L.n }, { 1 });
            else         b3[1] = mk(bb, RAD_BF16, 2, { L.rows, L.n }, { gu_p, 1 });
            b3[2] = mk(by, RAD_BF16, 2, { L.rows, L.n }, { L.n, 1 });
            memcpy(t, b3, sizeof(b3));
            RadArgs a{}; a.p = pv; a.n_p = 2; a.t = t; a.n_t = 3;
            rc = mul_e(&a, (RadStream)0);
            std::vector<uint16_t> oy((size_t)L.rows * L.n);
            if (rc == 0) { die("cy", hipMemcpy(oy.data(), ry, oy.size() * 2, hipMemcpyDeviceToHost));
                           hsh = fnv(hsh, oy.data(), oy.size() * 2); }
        }
        // red zones: every raw block's sentinel bands must be intact after the launch
        int rzbad = 0;
        {   std::vector<uint8_t> z(RZ * 2);
            for (size_t i = 0; i < raws.size(); ++i) {
                die("rzchk", hipMemcpy(z.data(), raws[i], RZ, hipMemcpyDeviceToHost));
                die("rzchk2", hipMemcpy(z.data() + RZ, (char*)raws[i] + RZ + 16 + szs[i], RZ, hipMemcpyDeviceToHost));
                for (uint8_t v : z) rzbad += v != 0xCDu;
                die("free", hipFree(raws[i])); } }
        printf("LEG %s rc=%d hash=%016llx rz=%d\n", L.id, rc, hsh, rzbad);
        die("syncL", hipDeviceSynchronize());
    }
    printf("PROBE DONE legs=%zu\n", sizeof(legs) / sizeof(Leg));
    return 0;
}
