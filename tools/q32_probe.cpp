// q32_probe.cpp -- correctness probe for the Q32 vectorised rmsnorm arm.
// Drives the shipped entry r3d_rad_rmsnorm through dlopen (the bench_bf16_gemv
// pattern) across the full leg matrix the queue row names: head dims, hidden
// dims, ragged, rejected-alignment, and pitch legs, each gain form, with a
// CPU reference (ref's ascending sum, same store chain), sentinel red-zones
// around every device buffer, and an FNV yhash per leg for cross-arm
// comparison (head/ragged/rejected legs MUST hash equal across arms -- a
// difference there means the gate leaked).
// One process = one lever state (R3D_RMS_VEC is read once); run twice and
// compare hashes. The R3D_DEBUG_RMS liveness line per launch lands on stderr
// in leg order; ARMWANT lines give the expected arm in the same order.
// Build/run (radiance-dev, gfx1100 pinned):
//   hipcc -O1 -std=c++20 -I/opt/radiance/include/radiance -Wall \
//     /work/tools/q32_probe.cpp -o /scratch/q32p -ldl
//   ROCR_VISIBLE_DEVICES=1 LIBR3D_SO=... R3D_RMS_VEC=0|1 /scratch/q32p
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "rad_abi.h"
#include "rad_types.h"

static void die(const char* m, hipError_t e) { if (e == hipSuccess) return;
    fprintf(stderr, "%s: %s\n", m, hipGetErrorName(e)); exit(2); }

static uint16_t f32_to_bf16(float f) {   // the shipped RTNE, host copy
    const uint32_t u = *reinterpret_cast<uint32_t*>(&f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float bf16_to_f32(uint16_t h) { const uint32_t u = (uint32_t)h << 16;
    float f; memcpy(&f, &u, 4); return f; }

static uint32_t st = 0x2468ace1u;
static uint32_t rnd() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; }
static uint16_t rbf() { return (uint16_t)((rnd() & 0x8000u) | 0x3F80u | (rnd() & 0x7fu)); }

struct Leg {
    const char* name;
    long n;            // row length
    long m;            // rows
    int  gain;         // 0 none, 1 bf16, 2 f32
    float wadd, mult;
    long x_off, y_off, w_off;   // base offsets in ELEMENTS (alignment legs)
    long x_pitch, y_pitch;      // 0 == n
};

static const float kEps = 1e-6f;
static constexpr long kGuard = 256;   // sentinel bytes each side

// guarded device buffer: 256B of 0xCD each side; verify() re-checks them.
struct Guard {
    void* base = nullptr; void* data = nullptr; size_t bytes = 0;
    void alloc(size_t n) {
        bytes = n; die("malloc", hipMalloc(&base, n + 2 * kGuard));
        data = (char*)base + kGuard;
        unsigned char pat[kGuard]; memset(pat, 0xcd, kGuard);
        die("padlo", hipMemcpy(base, pat, kGuard, hipMemcpyHostToDevice));
        die("padhi", hipMemcpy((char*)data + n, pat, kGuard, hipMemcpyHostToDevice));
    }
    bool verify() {
        unsigned char lo[kGuard], hi[kGuard];
        die("rdlo", hipMemcpy(lo, base, kGuard, hipMemcpyDeviceToHost));
        die("rdhi", hipMemcpy(hi, (char*)data + bytes, kGuard, hipMemcpyDeviceToHost));
        for (long i = 0; i < kGuard; ++i) if (lo[i] != 0xcd || hi[i] != 0xcd) return false;
        return true;
    }
    void free() { if (base) hipFree(base); }
};

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    setenv("R3D_DEBUG_RMS", "1", 1);   // liveness lines go to OUR stderr, leg order
    void* dl = dlopen(getenv("LIBR3D_SO") ? getenv("LIBR3D_SO")
                                          : "/build/radiance_home/kernels/libr3d.so",
                      RTLD_NOW | RTLD_LOCAL);
    if (!dl) { printf("dlopen libr3d: %s\n", dlerror()); return 1; }
    auto entry = (int (*)(const RadArgs*, RadStream))dlsym(dl, "r3d_rad_rmsnorm");
    if (!entry) { printf("dlsym r3d_rad_rmsnorm: %s\n", dlerror()); return 1; }

    // HARD device assertion (libr3d law): an unpinned arm produces phantom
    // passes on the wrong card, never a result. gfx1100 only, loudly.
    {
        hipDeviceProp_t prop;
        die("getprops", hipGetDeviceProperties(&prop, 0));
        printf("DEV name=%s arch=%s\n", prop.name, prop.gcnArchName);
        if (!strstr(prop.gcnArchName, "gfx1100")) {
            printf("PROBE-FAIL: not the XTX (arch=%s) -- ROCR pin is wrong\n", prop.gcnArchName);
            return 4;
        }
    }

    static const Leg legs[] = {
        // name              n     m  gain wadd mult  xoff yoff woff xpitch ypitch
        { "head128_none",    128,  8,   0, 0.f, 1.f,   0, 0, 0, 0, 0 },
        { "head256_bf16",    256,  8,   1, 0.f, 1.f,   0, 0, 0, 0, 0 },
        { "head256_f32",     256,  8,   2, 0.f, 1.f,   0, 0, 0, 0, 0 },
        { "gate1024_f32",   1024,  8,   2, 0.f, 1.f,   0, 0, 0, 0, 0 },
        { "hid2048_none",   2048,  8,   0, 0.f, 1.f,   0, 0, 0, 0, 0 },
        { "hid2048_bf16w",  2048,  8,   1, 1.f, 1.f,   0, 0, 0, 0, 0 },
        { "hid2048_f32m",   2048,  8,   2, 0.f, 0.5f,  0, 0, 0, 0, 0 },
        { "hid4096_none",   4096,  8,   0, 0.f, 1.f,   0, 0, 0, 0, 0 },
        { "hid4096_bf16",   4096,  8,   1, 0.f, 1.f,   0, 0, 0, 0, 0 },
        { "hid4096_f32",    4096,  8,   2, 0.f, 1.f,   0, 0, 0, 0, 0 },
        { "hid5120_bf16",   5120,  8,   1, 0.f, 1.f,   0, 0, 0, 0, 0 },
        { "hid6144_f32",    6144,  8,   2, 0.f, 1.f,   0, 0, 0, 0, 0 },
        { "ragged2047",     2047,  8,   1, 0.f, 1.f,   0, 0, 0, 0, 0 },
        { "ragged4095",     4095,  8,   0, 0.f, 1.f,   0, 0, 0, 0, 0 },
        { "align_x2",       4096,  8,   2, 0.f, 1.f,   2, 0, 0, 0, 0 },
        { "align_y2",       4096,  8,   1, 0.f, 1.f,   0, 2, 0, 0, 0 },
        { "align_w2",       4096,  8,   2, 0.f, 1.f,   0, 0, 2, 0, 0 },
        { "pitch2x",        4096,  8,   2, 0.f, 1.f,   0, 0, 0, 8192, 8192 },
        { "pitch_odd",      4096,  8,   1, 0.f, 1.f,   0, 0, 0, 4100, 4100 },
    };

    const int lever = getenv("R3D_RMS_VEC") ? atoi(getenv("R3D_RMS_VEC")) : 0;
    int fails = 0;
    for (const Leg& L : legs) {
        const long xp = L.x_pitch ? L.x_pitch : L.n;
        const long yp = L.y_pitch ? L.y_pitch : L.n;
        // the entry's gate, mirrored host-side (drives ARMWANT only; the shipped
        // gate is the authority, cross-checked by stderr liveness in the script)
        const bool want_vec = lever && L.n >= 2048 && (L.n % 8) == 0 && (xp % 8) == 0 &&
            (yp % 8) == 0 && ((L.x_off * 2) % 16) == 0 && ((L.y_off * 2) % 16) == 0 &&
            (L.gain == 0 || ((L.gain == 2 ? (L.w_off * 4) : (L.w_off * 2)) % 16) == 0);
        printf("ARMWANT leg=%s want=%s\n", L.name, want_vec ? "vec" : "scalar");

        Guard gx, gy, gw;
        const long x_el = L.x_off + (L.m - 1) * xp + L.n;
        const long y_el = L.y_off + (L.m - 1) * yp + L.n;
        gx.alloc(x_el * 2); gy.alloc(y_el * 2);
        if (L.gain) gw.alloc(L.n * (L.gain == 2 ? 4 : 2) + (L.gain == 2 ? L.w_off * 4 : L.w_off * 2));

        std::vector<uint16_t> hx(x_el, 0), hy_w(y_el);
        for (auto& v : hx) v = rbf();
        die("copyx", hipMemcpy(gx.data, hx.data(), x_el * 2, hipMemcpyHostToDevice));

        const bool wf32 = L.gain == 2;
        std::vector<float> hwf; std::vector<uint16_t> hwb;
        void* wp = nullptr;
        if (L.gain) {
            if (wf32) { hwf.resize(L.n); for (auto& v : hwf) v = 1.0f + (float)(rnd() & 0xffff) / 65536.0f; }
            else      { hwb.resize(L.n); for (auto& v : hwb) v = rbf(); }
            wp = (char*)gw.data + (wf32 ? L.w_off * 4 : L.w_off * 2);
            die("copyw", hipMemcpy(wp, wf32 ? (const void*)hwf.data() : (const void*)hwb.data(),
                                   L.n * (wf32 ? 4 : 2), hipMemcpyHostToDevice));
        }

        auto mk = [](void* d, uint32_t dt, int rank, std::vector<int64_t> sh,
                     std::vector<int64_t> stride) {
            RadTensor x{}; x.data = d; x.dtype = dt; x.rank = (uint32_t)rank;
            for (size_t i = 0; i < sh.size(); ++i) { x.shape[i] = sh[i]; x.stride[i] = stride[i]; }
            return x; };
        RadTensor t[3] = {
            mk((char*)gx.data + L.x_off * 2, RAD_BF16, 2, { L.m, L.n }, { xp, 1 }),
            mk(wp, wf32 ? RAD_F32 : RAD_BF16, 1, { L.n }, { 1 }),
            mk((char*)gy.data + L.y_off * 2, RAD_BF16, 2, { L.m, L.n }, { yp, 1 }),
        };
        if (!L.gain) t[1].data = nullptr;
        RadParam pv[] = { RAD_INT("n", L.n) };
        RadArgs a{}; a.p = pv; a.n_p = 1; a.t = t; a.n_t = 3;
        const int rc = entry(&a, (RadStream)0);
        die("sync", hipDeviceSynchronize());

        std::vector<uint16_t> hy(y_el, 0);
        die("copyy", hipMemcpy(hy.data(), gy.data, y_el * 2, hipMemcpyDeviceToHost));

        // reference: ref's ascending f32 sum, device's store chain, per leg
        double num = 0.0, den = 0.0, maxulp = 0.0;
        for (long r = 0; r < L.m; ++r) {
            float ss = 0.0f;
            for (long i = 0; i < L.n; ++i) {
                const float v = bf16_to_f32(hx[L.x_off + r * xp + i]);
                ss += v * v;
            }
            const float s = 1.0f / sqrtf(ss / (float)L.n + kEps);
            for (long i = 0; i < L.n; ++i) {
                const float g = !L.gain ? 1.0f : wf32 ? hwf[i] : bf16_to_f32(hwb[i]);
                const float e = bf16_to_f32(hx[L.x_off + r * xp + i]) * s * (g + L.wadd) * L.mult;
                const uint16_t want = f32_to_bf16(e);
                const uint16_t got  = hy[L.y_off + r * yp + i];
                const float wf = bf16_to_f32(want), gf = bf16_to_f32(got);
                num += (double)(gf - wf) * (gf - wf); den += (double)wf * wf;
                const float mag = fabsf(wf) > 1e-30f ? fabsf(wf) : 1e-30f;
                const double ulp = fabs((double)gf - wf) / (mag * 0.0078125);   // bf16 7-bit ulp
                if (ulp > maxulp) maxulp = ulp;
            }
        }
        const float rel_l2 = den > 0 ? (float)sqrt(num / den) : 0.0f;
        bool redz = gx.verify() && gy.verify(); if (L.gain) redz = redz && gw.verify();

        unsigned long long yh = 1469598103934665603ull;   // FNV-1a over the y REGION
        for (long r = 0; r < L.m; ++r)
            for (long i = 0; i < L.n; ++i) {
                const uint16_t v = hy[L.y_off + r * yp + i];
                yh = (yh ^ (v & 0xffu)) * 1099511628211ull;
                yh = (yh ^ (v >> 8))    * 1099511628211ull;
            }
        const bool pass = rc == 0 && redz && rel_l2 < 2e-3 && maxulp <= 4.0;
        if (!pass) ++fails;
        printf("PROBE leg=%s rc=%d rel_l2=%.3e max_ulp=%.1f yhash=%016llx redz=%s %s\n",
               L.name, rc, rel_l2, maxulp, yh, redz ? "OK" : "BAD", pass ? "ok" : "FAIL");
        fflush(stdout);
        gx.free(); gy.free(); gw.free();
    }
    if (fails) printf("PROBE-FAIL (%d legs)\n", fails);
    else       printf("PROBE-OK-MATRIX\n");
    return fails ? 1 : 0;
}
