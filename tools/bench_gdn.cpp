/* bench_gdn -- Q20 decode-shape timing for the GDN recurrent rows.
 *
 * Decode is where GDN lives (48 of 64 layers of the Flash-Next hybrid every step),
 * so the cases mirror decode traffic: slen=1 per sequence, rotating state slots
 * (a slot manager hands fresh slots every step -- an unrotated bench measures a
 * pinned slice the engine never gets to reuse), the anchor fold replaying the
 * previous step's accepted tokens, and the live spec band's quantiser pair on.
 *
 * Same-binary A/B: the R3D_GDN_SB env lever picks the instantiation at launch,
 * so baseline/fenced/fused arms are one process apart only by the env -- no
 * build-to-build drift. Every case prints a fingerprint (hash of o + touched
 * state slots) after the timing loop: any lever that changes it is NOT a
 * scheduling-only change (SB arms must hash-equal at equal case/seed).
 *
 * usage: bench_gdn <libr3d.so> [--iters N] [--only case-substr]
 * run inside the device container with ROCR_VISIBLE_DEVICES=1.
 */
#include <hip/hip_runtime.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "rad_abi.h"

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) {                        \
    std::printf("HIP FAIL %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_));          \
    std::exit(1); } } while (0)

typedef int (*CountFn)();
typedef const RadKernelInfo* (*AtFn)(int);

static const RadKernelInfo* g_row = nullptr;
static void row_open(const char* path, const char* op) {
    void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::printf("dlopen %s: %s\n", path, dlerror()); std::exit(1); }
    auto count = (CountFn)dlsym(h, "rad_kernel_count");
    auto at    = (AtFn)dlsym(h, "rad_kernel_at");
    if (!count || !at) { std::printf("dlsym failed\n"); std::exit(1); }
    for (int i = 0; i < count(); ++i) {
        const RadKernelInfo* k = at(i);
        if (k && std::strcmp(k->op, op) == 0 && k->domain == RAD_DOMAIN_DEVICE) { g_row = k; return; }
    }
    std::printf("row %s not found\n", op); std::exit(1);
}

static int dt_bytes(uint32_t d) {
    if (d == RAD_F8E4M3 || d == RAD_F8E5M2 || d == RAD_I8 || d == RAD_U8) return 1;
    return (d == RAD_F32 || d == RAD_I32) ? 4 : 2;
}

struct Buf {
    std::vector<uint8_t> h;
    void* d = nullptr; size_t cap = 0;
    uint32_t dtype = 0;
    std::vector<int64_t> shape;
    size_t n() const { size_t r = 1; for (auto s : shape) r *= (size_t)s; return r; }
    void init(uint32_t dt, std::vector<int64_t> sh) {
        dtype = dt; shape = std::move(sh);
        size_t bytes = n() * dt_bytes(dt); if (!bytes) bytes = 16;
        h.assign(bytes, 0);
    }
    void alloc_dev() { if (cap < h.size()) { if (d) hipFree(d); CK(hipMalloc(&d, h.size())); cap = h.size(); } }
    void to_dev()  { alloc_dev(); CK(hipMemcpy(d, h.data(), h.size(), hipMemcpyHostToDevice)); }
    ~Buf() { if (d) hipFree(d); }
};

static void mk_tensor(RadTensor& t, Buf& b) {
    t.data = b.d; t.dtype = b.dtype; t.rank = (uint32_t)b.shape.size();
    int64_t s = 1;
    for (int i = (int)b.shape.size() - 1; i >= 0; --i) {
        t.shape[i] = b.shape[i]; t.stride[i] = s; s *= b.shape[i];
    }
}

static void fill_bf16(std::mt19937& rng, Buf& b, float amp) {
    std::uniform_real_distribution<float> u(-amp, amp);
    uint16_t* p = (uint16_t*)b.h.data();
    for (size_t i = 0; i < b.n(); ++i) {
        float f = u(rng) * amp; uint32_t u32; std::memcpy(&u32, &f, 4);
        p[i] = (uint16_t)(u32 >> 16);
    }
}
static void fill_f32(std::mt19937& rng, Buf& b, float amp) {
    std::uniform_real_distribution<float> u(-amp, amp);
    float* p = (float*)b.h.data();
    for (size_t i = 0; i < b.n(); ++i) p[i] = u(rng) * amp;
}
static void set_i32(Buf& b, std::initializer_list<int32_t> v) {
    size_t i = 0;
    for (int32_t x : v) { if (i >= b.n()) break; ((int32_t*)b.h.data())[i++] = x; }
}

/* FNV-1a over the output buffer + the state pool: equal across SB arms, else the
 * lever changed math, not schedule. */
static uint64_t fnv(const std::vector<uint8_t>& v) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t c : v) { h ^= c; h *= 1099511628211ull; }
    return h;
}

struct Case {
    const char* name; int fused; int anchor; int N;
    int iters;
};

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s <libr3d.so> [--iters N] [--only substr]\n", argv[0]); return 2; }
    setvbuf(stdout, nullptr, _IONBF, 0);
    int iters = 300;
    const char* only = nullptr;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
    }
    CK(hipSetDevice(0));

    const int K = 128, V = (getenv("BG_V") ? atoi(getenv("BG_V")) : 128), Hv = 32, Hg = 8, GRP = Hv / Hg; /* Q52: BG_V lever for ragged-width determinism forensics; default = shipped geometry */
    const int SLOTS = 64, SW = 6;
    const int CD = 2 * Hg * K + Hv * V, CSL = 6;

    std::mt19937 rng(20261027);
    const char* rec_op  = "gdn_recurrent_update";
    const char* fus_op  = "gdn_conv_recurrent_update";

    /* one library per process keeps each row's own tensor list simple */
    Case cases[] = {
        { "rec_pc_n1", 0, 0, 1, iters }, { "rec_pc_n8", 0, 0, 8, iters },
        { "rec_an_n1", 0, 1, 1, iters }, { "rec_an_n8", 0, 1, 8, iters },
        { "fus_an_n1", 1, 1, 1, iters }, { "fus_an_n8", 1, 1, 8, iters },
    };

    for (const Case& C : cases) {
        if (only && !std::strstr(C.name, only)) continue;
        row_open(argv[1], C.fused ? fus_op : rec_op);

        const int N = C.N, T = N;                    /* decode: slen 1 per seq */
        std::vector<Buf> o(C.fused ? 18 : 16);
        auto A = [&](int i, uint32_t dt, std::vector<int64_t> sh) { o[i].init(dt, std::move(sh)); };
        std::vector<RadParam> p;
        auto pi = [&](const char* k, long long v) { p.push_back(RadParam{k, RAD_P_INT, v, 0, nullptr, 0}); };
        auto pf = [&](const char* k, double v)    { p.push_back(RadParam{k, RAD_P_F64, 0, 0, nullptr, v}); };
        auto ps = [&](const char* k, const char* v) { p.push_back(RadParam{k, RAD_P_STR, 0, 0, v, 0}); };
        pi("head_k", K); pi("head_v", V); pi("n_head_v", Hv);
        if (C.fused) pi("conv_width", 4);
        pf("scale", 0.08838834764831845); pf("eps", 1e-6); pf("l2_eps", 1e-6);
        pf("softplus_thr", 20.0); ps("act", "silu"); pi("state_fp16", 0);
        ps("state_form", C.anchor ? "anchor" : "per_candidate");

        int idx = 0, i_x = -1, i_w = -1, i_cb = -1, i_cs = -1, i_ci = -1;
        int i_q, i_k, i_v, i_av, i_bv, i_al, i_db, i_st, i_si, i_na, i_cu, i_z, i_nw, i_o, i_oq, i_os;
        if (C.fused) {
            i_x = idx; A(idx++, RAD_BF16, {T, CD});
            i_w = idx; A(idx++, RAD_BF16, {CD, 4});
            i_cb = idx; A(idx++, RAD_F32, {CD});
            i_cs = idx; A(idx++, RAD_F32, {SLOTS, CD, CSL});
            i_ci = idx; A(idx++, RAD_I32, {N});
        } else {
            i_q = idx; A(idx++, RAD_BF16, {T, Hg, K});
            i_k = idx; A(idx++, RAD_BF16, {T, Hg, K});
            i_v = idx; A(idx++, RAD_BF16, {T, Hv, V});
        }
        i_av = idx; A(idx++, RAD_BF16, {T, Hv});
        i_bv = idx; A(idx++, RAD_BF16, {T, Hv});
        i_al = idx; A(idx++, RAD_F32, {Hv});
        i_db = idx; A(idx++, RAD_F32, {Hv});
        i_st = idx; A(idx++, RAD_F32, {SLOTS, Hv, V, K});
        i_si = idx; A(idx++, RAD_I32, {N, SW});
        i_na = idx; A(idx++, RAD_I32, {N});
        i_cu = idx; A(idx++, RAD_I32, {N + 1});
        i_z  = idx; A(idx++, RAD_BF16, {T, Hv, V});
        i_nw = idx; A(idx++, RAD_F32, {V});
        i_o  = idx; A(idx++, RAD_BF16, {T, Hv, V});
        if (C.anchor) { i_oq = idx; A(idx++, RAD_F8E4M3, {T, Hv, V});
                        i_os = idx; A(idx++, RAD_F32, {T, Hv}); }
        else          { i_oq = idx; A(idx++, RAD_F32, {1}); o[i_oq].h.assign(4, 0);
                        i_os = idx; A(idx++, RAD_F32, {1}); o[i_os].h.assign(4, 0); }
        pi("q_len", T);

        /* seeds: the pool ZEROED as a live boot hands it (a random pool replays
         * NaN-yield garbage no engine feeds); everything else realistic-small */
        if (C.fused) { fill_bf16(rng, o[i_x], 1.5f); fill_bf16(rng, o[i_w], 1.0f);
                       fill_f32(rng, o[i_cb], 0.5f); fill_f32(rng, o[i_cs], 2.0f); }
        else         { fill_bf16(rng, o[i_q], 1.0f); fill_bf16(rng, o[i_k], 1.0f);
                       fill_bf16(rng, o[i_v], 2.0f); }
        fill_bf16(rng, o[i_av], 4.0f); fill_bf16(rng, o[i_bv], 6.0f);
        fill_f32(rng, o[i_al], 3.0f);  fill_f32(rng, o[i_db], 1.0f);
        fill_f32(rng, o[i_nw], 1.0f);  fill_bf16(rng, o[i_z], 2.0f);
        for (int n = 0; n < N; ++n) ((int32_t*)o[i_cu].h.data())[n] = n;
        ((int32_t*)o[i_cu].h.data())[N] = N;
        for (int n = 0; n < N; ++n) ((int32_t*)o[i_na].h.data())[n] = C.anchor ? 2 : 1;
        for (auto& b : o) b.to_dev();

        std::vector<RadTensor> th(o.size());
        for (size_t i = 0; i < o.size(); ++i) mk_tensor(th[i], o[i]);
        /* absent quantiser pair for per_candidate: a dummy 1-elem f32 (the row's
         * optional-tensor convention, as test_anchor_chain's present=false path) */
        if (!C.anchor) for (int i : {i_oq, i_os}) {
            std::memset(&th[i], 0, sizeof th[i]);
            th[i].rank = 1; th[i].shape[0] = 1; th[i].stride[0] = 1; th[i].dtype = RAD_F32;
        }

        /* rotating slot assignment: recompute the sidx contents per launch
         * (outside the event window) exactly like a slot manager would. The
         * 6-column window of every sequence must stay DISJOINT (blocks writing
         * aliased slots race and the fingerprint would wobble): with SLOTS=64
         * and bases under 16, base + n*6 + c never wraps for N <= 8. */
        auto slots_for = [&](int base) {
            int32_t* si = (int32_t*)o[i_si].h.data();
            for (int n = 0; n < N; ++n) for (int c = 0; c < SW; ++c)
                si[n * SW + c] = base + n * SW + c;
            if (C.fused) { int32_t* ci = (int32_t*)o[i_ci].h.data();
                           for (int n = 0; n < N; ++n) ci[n] = (base / 2 + n) % SLOTS; }
        };

        std::vector<float> ts;
        hipEvent_t ev0, ev1; CK(hipEventCreate(&ev0)); CK(hipEventCreate(&ev1));
        int rot = C.anchor ? 3 : 8;
        for (int it = -30; it < C.iters; ++it) {
            slots_for(((it % rot) + rot) % rot * (C.anchor ? 4 : 2));
            o[i_si].to_dev(); if (C.fused) o[i_ci].to_dev();
            RadArgs ad{}; ad.t = th.data(); ad.n_t = (int)o.size();
            ad.p = p.data(); ad.n_p = (int)p.size();
            CK(hipDeviceSynchronize());
            CK(hipEventRecord(ev0, (hipStream_t)nullptr));
            int rc = g_row->launch(&ad, (RadStream)nullptr);
            if (rc != RAD_OK) { std::printf("%s LAUNCH rc=%d\n", C.name, rc); return 1; }
            CK(hipEventRecord(ev1, (hipStream_t)nullptr));
            CK(hipEventSynchronize(ev1));
            if (it >= 0) { float ms = 0; CK(hipEventElapsedTime(&ms, ev0, ev1)); ts.push_back(ms * 1000.0f); }
        }
        CK(hipMemcpy(o[i_o].h.data(), o[i_o].d, o[i_o].h.size(), hipMemcpyDeviceToHost));
        CK(hipMemcpy(o[i_st].h.data(), o[i_st].d, o[i_st].h.size(), hipMemcpyDeviceToHost));
        std::sort(ts.begin(), ts.end());
        uint64_t fp = fnv(o[i_o].h) * 1000003ull + fnv(o[i_st].h);
        std::printf("%-10s med=%8.1fus p25=%8.1f p75=%8.1f min=%8.1f n=%zu fp=%016llx\n",
                    C.name, ts[ts.size()/2], ts[ts.size()/4], ts[ts.size()*3/4], ts.front(),
                    ts.size(), (unsigned long long)fp);
        CK(hipEventDestroy(ev0)); CK(hipEventDestroy(ev1));
    }
    return 0;
}
