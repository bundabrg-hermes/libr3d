// q49_bench.cpp -- Q49 attribution + A/B bench for the elementwise family:
// gated_quant_fp8 (MLP act-mul+quant), mul (actmul attn-gate), quant_act_fp8
// (quant sibling), cast (bf16->bf16), and the vec rmsnorm anchor legs.
// Laws from bench_rmsnorm (Q32): batch timing, rot planes for cold streams,
// random operands + FNV yhash liveness per leg.
// Build: hipcc -O1 -std=c++20 -I/opt/radiance/include/radiance -Wall q49_bench.cpp -o q49b -ldl
// Run:   ROCR_VISIBLE_DEVICES=1 LIBR3D_SO=... ./q49b <kernel> <M> <reps> <batch> <rot>
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
static uint32_t st = 0x2468ace1u;
static uint32_t rnd() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; }
static uint16_t rbf() { return (uint16_t)((rnd() & 0x8000u) | 0x3F80u | (rnd() & 0x7fu)); }

static RadTensor mk(void* d, uint32_t dt, int rank, std::vector<int64_t> sh,
                    std::vector<int64_t> stride) {
    RadTensor x{}; x.data = d; x.dtype = dt; x.rank = (uint32_t)rank;
    for (size_t i = 0; i < sh.size(); ++i) { x.shape[i] = sh[i]; x.stride[i] = stride[i]; }
    return x;
}
static void* dall(size_t bytes) { void* p; die("malloc", hipMalloc(&p, bytes)); return p; }
static void* fillb(size_t n, int u16) {           // device buffer with random bf16/bytes
    std::vector<uint8_t> h(n);
    if (u16) { auto* p = (uint16_t*)h.data(); for (size_t i = 0; i < n / 2; ++i) p[i] = rbf(); }
    else for (auto& v : h) v = (uint8_t)rnd();
    void* d = dall(n); die("copy", hipMemcpy(d, h.data(), n, hipMemcpyHostToDevice)); return d;
}
static unsigned long long fnv(const uint8_t* p, size_t n) {
    unsigned long long h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: q49b <gate|mul|qact|cast|rms> M reps batch rot [n]\n"); return 2; }
    const std::string k = argv[1];
    const long M = atol(argv[2]);
    const int reps = argc > 3 ? atoi(argv[3]) : 30, batch = argc > 4 ? atoi(argv[4]) : 20;
    const int rot = argc > 5 ? atoi(argv[5]) : 1;
    void* dl = dlopen(getenv("LIBR3D_SO") ? getenv("LIBR3D_SO") : "libr3d.so", RTLD_NOW | RTLD_LOCAL);
    if (!dl) { printf("dlopen: %s\n", dlerror()); return 1; }
    typedef int (*fn_t)(const RadArgs*, RadStream);
    auto need = [&](const char* s) { auto f = (fn_t)dlsym(dl, s);
        if (!f) { printf("dlsym %s: %s\n", s, dlerror()); exit(1); } return f; };

    RadTensor* t = nullptr; die("hostT", hipHostMalloc((void**)&t, sizeof(RadTensor) * 8));

    // ---- per-kernel setups ------------------------------------------------
    if (k == "gate") {                       // gated_quant_fp8: n=12288 group=128
        const long n = argc > 6 ? atol(argv[6]) : 12288, g = 128;
        auto entry = need("r3d_rad_gated_quant_fp8");
        const size_t gub = (size_t)M * 2 * n * 2, qb = (size_t)M * n, sb = (size_t)M * (n / g) * 4;
        std::vector<void*> dgu(rot); for (int p = 0; p < rot; ++p) dgu[p] = fillb(gub, 1);
        void* dq = dall(qb); void* dsc = dall(sb);
        RadParam pv[] = { RAD_INT("n", n), RAD_INT("group", g), RAD_STR("act", "silu") };
        double bytes = (double)(gub + qb + sb);
        auto launch = [&](int p) {
            RadTensor b[3] = { mk(dgu[p], RAD_BF16, 2, { M, 2 * n }, { 2 * n, 1 }),
                               mk(dq, RAD_F8E4M3, 2, { M, n }, { n, 1 }),
                               mk(dsc, RAD_F32, 2, { M, n / g }, { n / g, 1 }) };
            memcpy(t, b, sizeof(b));
            RadArgs a{}; a.p = pv; a.n_p = 3; a.t = t; a.n_t = 3;
            return entry(&a, (RadStream)0); };
        if (launch(0) != 0) { printf("gate ENTRY REFUSED\n"); return 3; }
        die("warm", hipDeviceSynchronize());
        std::vector<double> tms;
        for (int r = 0; r < reps; ++r) { timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int bb = 0; bb < batch; ++bb) launch((r * batch + bb) % rot);
            die("sync", hipDeviceSynchronize()); clock_gettime(CLOCK_MONOTONIC, &t1);
            tms.push_back(((t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec)) / batch); }
        std::sort(tms.begin(), tms.end()); double med = tms[tms.size() / 2];
        std::vector<uint8_t> hq(qb); die("cpy", hipMemcpy(hq.data(), dq, qb, hipMemcpyDeviceToHost));
        printf("ROW gate M=%ld n=%ld med_ms=%.6f gbs=%.1f yhash=%016llx\n",
               M, n, med * 1e3, bytes / med / 1e9, fnv(hq.data(), qb));
        return 0;
    }
    if (k == "mul" || k == "cast") {
        const long n = argc > 6 ? atol(argv[6]) : 4096;
        auto entry = need(k == "mul" ? "r3d_rad_mul" : "r3d_rad_cast");
        std::vector<void*> da(rot), db(rot);
        for (int p = 0; p < rot; ++p) { da[p] = fillb((size_t)M * n * 2, 1); db[p] = fillb((size_t)M * n * 2, 1); }
        void* dy = dall((size_t)M * n * 2);
        RadParam pv[] = { RAD_INT("n", n), RAD_STR("dtype", "bf16") };
        double bytes = (k == "mul") ? 3.0 * M * n * 2 : 2.0 * M * n * 2;
        auto launch = [&](int p) {
            RadTensor b[3] = { mk(da[p], RAD_BF16, 2, { M, n }, { n, 1 }),
                               mk(db[p], RAD_BF16, 2, { M, n }, { n, 1 }),
                               mk(dy, RAD_BF16, 2, { M, n }, { n, 1 }) };
            memcpy(t, b, sizeof(b));
            RadArgs a{}; a.p = pv; a.n_p = 2; a.t = t; a.n_t = (k == "mul") ? 3 : 2;
            return entry(&a, (RadStream)0); };
        if (launch(0) != 0) { printf("%s ENTRY REFUSED\n", k.c_str()); return 3; }
        die("warm", hipDeviceSynchronize());
        std::vector<double> tms;
        for (int r = 0; r < reps; ++r) { timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int bb = 0; bb < batch; ++bb) launch((r * batch + bb) % rot);
            die("sync", hipDeviceSynchronize()); clock_gettime(CLOCK_MONOTONIC, &t1);
            tms.push_back(((t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec)) / batch); }
        std::sort(tms.begin(), tms.end()); double med = tms[tms.size() / 2];
        std::vector<uint8_t> hy((size_t)M * n * 2); die("cpy", hipMemcpy(hy.data(), dy, hy.size(), hipMemcpyDeviceToHost));
        printf("ROW %s M=%ld n=%ld med_ms=%.6f gbs=%.1f yhash=%016llx\n",
               k.c_str(), M, n, med * 1e3, bytes / med / 1e9, fnv(hy.data(), hy.size()));
        return 0;
    }
    if (k == "qact") {
        const long n = argc > 6 ? atol(argv[6]) : 4096, g = 128;
        auto entry = need("r3d_rad_quant_act_fp8");
        std::vector<void*> dx(rot); for (int p = 0; p < rot; ++p) dx[p] = fillb((size_t)M * n * 2, 1);
        void* dq = dall((size_t)M * n); void* dsc = dall((size_t)M * (n / g) * 4);
        RadParam pv[] = { RAD_INT("n", n), RAD_INT("group", g) };
        double bytes = M * n * 2 + M * n + M * (n / g) * 4.0;
        auto launch = [&](int p) {
            RadTensor b[3] = { mk(dx[p], RAD_BF16, 2, { M, n }, { n, 1 }),
                               mk(dq, RAD_F8E4M3, 2, { M, n }, { n, 1 }),
                               mk(dsc, RAD_F32, 2, { M, n / g }, { n / g, 1 }) };
            memcpy(t, b, sizeof(b));
            RadArgs a{}; a.p = pv; a.n_p = 2; a.t = t; a.n_t = 3;
            return entry(&a, (RadStream)0); };
        if (launch(0) != 0) { printf("qact ENTRY REFUSED\n"); return 3; }
        die("warm", hipDeviceSynchronize());
        std::vector<double> tms;
        for (int r = 0; r < reps; ++r) { timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int bb = 0; bb < batch; ++bb) launch((r * batch + bb) % rot);
            die("sync", hipDeviceSynchronize()); clock_gettime(CLOCK_MONOTONIC, &t1);
            tms.push_back(((t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec)) / batch); }
        std::sort(tms.begin(), tms.end()); double med = tms[tms.size() / 2];
        std::vector<uint8_t> hq((size_t)M * n); die("cpy", hipMemcpy(hq.data(), dq, hq.size(), hipMemcpyDeviceToHost));
        printf("ROW qact M=%ld n=%ld med_ms=%.6f gbs=%.1f yhash=%016llx\n",
               M, n, med * 1e3, bytes / med / 1e9, fnv(hq.data(), hq.size()));
        return 0;
    }
    if (k == "rms") {                        // anchor: shipped rmsnorm (vec-gated n>=2048)
        const long n = argc > 6 ? atol(argv[6]) : 4096;
        auto entry = need("r3d_rad_rmsnorm");
        std::vector<void*> dx(rot); for (int p = 0; p < rot; ++p) dx[p] = fillb((size_t)M * n * 2, 1);
        void* dy = dall((size_t)M * n * 2); void* dw = dall((size_t)n * 2);
        { std::vector<uint16_t> h(n); for (auto& v : h) v = rbf();
          die("cw", hipMemcpy(dw, h.data(), (size_t)n * 2, hipMemcpyHostToDevice)); }
        RadParam pv[] = { RAD_INT("n", n) };
        double bytes = 2.0 * M * n * 2 + n * 2;
        auto launch = [&](int p) {
            RadTensor b[3] = { mk(dx[p], RAD_BF16, 2, { M, n }, { n, 1 }),
                               mk(dw, RAD_BF16, 1, { n }, { 1 }),
                               mk(dy, RAD_BF16, 2, { M, n }, { n, 1 }) };
            memcpy(t, b, sizeof(b));
            RadArgs a{}; a.p = pv; a.n_p = 1; a.t = t; a.n_t = 3;
            return entry(&a, (RadStream)0); };
        if (launch(0) != 0) { printf("rms ENTRY REFUSED\n"); return 3; }
        die("warm", hipDeviceSynchronize());
        std::vector<double> tms;
        for (int r = 0; r < reps; ++r) { timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int bb = 0; bb < batch; ++bb) launch((r * batch + bb) % rot);
            die("sync", hipDeviceSynchronize()); clock_gettime(CLOCK_MONOTONIC, &t1);
            tms.push_back(((t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec)) / batch); }
        std::sort(tms.begin(), tms.end()); double med = tms[tms.size() / 2];
        std::vector<uint8_t> hy((size_t)M * n * 2); die("cpy", hipMemcpy(hy.data(), dy, hy.size(), hipMemcpyDeviceToHost));
        printf("ROW rms M=%ld n=%ld med_ms=%.6f gbs=%.1f yhash=%016llx\n",
               M, n, med * 1e3, bytes / med / 1e9, fnv(hy.data(), hy.size()));
        return 0;
    }
    printf("unknown kernel %s\n", argv[1]); return 2;
}
