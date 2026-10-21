/* test_conv_pair.cpp -- differential unit test for libr3d's conv pair (gdn_conv_prep,
 * gdn_conv_recurrent_update) against libref's own host rows, byte for byte.
 *
 * Why a unit test and not kbench: no pack records these two ops -- the ssm-dt dtype quirk
 * (ssm_dt.bias/ssm_a_log stored bf16, spec f32) declines them at record time, so the packs
 * carry zero matching cases. This tool dlopens BOTH plugins, finds each row through the
 * ordinary kernel table (the same table the engine selects from), launches both on
 * identical synthetic inputs at 27B-band geometries, and requires the output bytes -- and
 * the INOUT states -- to match exactly.
 *
 * Flow per case: seed every operand; run the DEVICE row on device copies; pull its outputs;
 * restore the seeds; run the REF row on host; compare byte for byte. INOUT operands are
 * compared too, because a conv/recurrent row is only correct if its state rewrite matches.
 *
 * Build (in radiance-dev, on the card):
 *   hipcc -O2 --offload-arch=gfx1100 -I/opt/radiance/include/radiance \
 *       tools/test_conv_pair.cpp -ldl -o /tmp/test_conv_pair
 * Run:
 *   /tmp/test_conv_pair <libr3d.so> <libref.so>
 */
#include <hip/hip_runtime.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "rad_abi.h"

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) {                        \
    std::printf("HIP FAIL %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_));          \
    std::exit(1); } } while (0)

typedef int (*CountFn)();
typedef const RadKernelInfo* (*AtFn)(int);

struct Lib {
    void* h = nullptr; CountFn count = nullptr; AtFn at = nullptr;
    void open(const char* path) {
        h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!h) { std::printf("dlopen %s: %s\n", path, dlerror()); std::exit(1); }
        count = (CountFn)dlsym(h, "rad_kernel_count");
        at    = (AtFn)dlsym(h, "rad_kernel_at");
        if (!count || !at) { std::printf("dlsym failed in %s\n", path); std::exit(1); }
    }
    const RadKernelInfo* row(const char* op, int domain) {
        for (int i = 0; i < count(); ++i) {
            const RadKernelInfo* k = at(i);
            if (k && std::strcmp(k->op, op) == 0 && k->domain == domain) return k;
        }
        return nullptr;
    }
};

int dt_bytes(uint32_t d) {
    if (d == RAD_F8E4M3 || d == RAD_F8E5M2 || d == RAD_I8 || d == RAD_U8) return 1;
    return (d == RAD_F32 || d == RAD_I32) ? 4 : 2;
}

struct Buf {
    std::vector<uint8_t> h, seed, dout;
    void* d = nullptr;
    uint32_t dtype = 0;
    std::vector<int64_t> shape;
    size_t n() const { size_t r = 1; for (auto s : shape) r *= (size_t)s; return r; }
    void init(uint32_t dt, std::vector<int64_t> sh) {
        dtype = dt; shape = std::move(sh);
        size_t bytes = n() * dt_bytes(dt); if (!bytes) bytes = 16;
        h.assign(bytes, 0); seed.assign(bytes, 0); dout.assign(bytes, 0);
    }
    void snapshot() { seed = h; }
    void restore()  { h = seed; }
    void alloc_dev() { if (!d) CK(hipMalloc(&d, h.size())); }
    void to_dev()  { alloc_dev(); CK(hipMemcpy(d, h.data(), h.size(), hipMemcpyHostToDevice)); }
    ~Buf() { if (d) hipFree(d); }
};

static void mk_tensor(RadTensor& t, Buf& b, void* data) {
    t.data = data; t.dtype = b.dtype; t.rank = (uint32_t)b.shape.size();
    int64_t s = 1;
    for (int i = (int)b.shape.size() - 1; i >= 0; --i) {
        t.shape[i] = b.shape[i]; t.stride[i] = s; s *= b.shape[i];
    }
}

static void fill_f32(std::mt19937& rng, Buf& b, float amp) {
    std::uniform_real_distribution<float> u(-amp, amp);
    float* p = (float*)b.h.data();
    for (size_t i = 0; i < b.n(); ++i) p[i] = u(rng) * amp;
}
static void fill_bf16(std::mt19937& rng, Buf& b, float amp) {
    std::uniform_real_distribution<float> u(-amp, amp);
    uint16_t* p = (uint16_t*)b.h.data();
    for (size_t i = 0; i < b.n(); ++i) {
        float f = u(rng) * amp; uint32_t u32; std::memcpy(&u32, &f, 4);
        p[i] = (uint16_t)(u32 >> 16);            // truncating round: any bits are legal input
    }
}
static void fill_i32(std::initializer_list<int32_t> v, Buf& b) {
    size_t i = 0;
    for (int32_t x : v) { if (i >= b.n()) break; ((int32_t*)b.h.data())[i++] = x; }
}

static int g_fail = 0;

/* One operand slot: buffer + role (0 IN, 1 OUT, 2 INOUT) + name. A null-data operand is a
 * tensor slot with rank 1 shape {1} and data == nullptr: optional operands the row skips. */
struct Opd { Buf b; int kind = 0; const char* name = ""; bool present = true; };

struct Case {
    std::vector<Opd> o;
    std::vector<RadParam> p;
    std::vector<int> outs;                       // operand indices to compare
    const char* title = "";

    int add(int kind, const char* name, uint32_t dt, std::vector<int64_t> sh) {
        o.push_back(Opd{}); o.back().kind = kind; o.back().name = name;
        o.back().b.init(dt, std::move(sh));
        return (int)o.size() - 1;
    }
    void pi(const char* k, long long v) { p.push_back(RadParam{k, RAD_P_INT, v, 0, nullptr, 0}); }
    void pf(const char* k, double v)    { p.push_back(RadParam{k, RAD_P_F64, 0, 0, nullptr, v}); }
    void ps(const char* k, const char* v) { p.push_back(RadParam{k, RAD_P_STR, 0, 0, v, 0}); }
};

static void compare(const Opd& a, const Opd& d) {
    if (std::memcmp(a.b.h.data(), d.b.dout.data(), a.b.h.size()) == 0) return;
    ++g_fail;
    size_t eb = dt_bytes(a.b.dtype), bad = 0, shown = 0;
    const auto& sh = a.b.shape;
    /* Histogram by (dim0, dim1) cell: rows of a [T,H,.] output or [slots,H,.] state are
     * exactly one head, and knowing WHICH heads diverge is the diagnosis. */
    size_t cell = sh.size() >= 3 ? (size_t)sh[sh.size() - 2] * (size_t)sh[sh.size() - 1] : 1;
    std::vector<std::pair<size_t, size_t>> hist;   // (cell index, count)
    for (size_t i = 0; i < a.b.n(); ++i) {
        const uint8_t* pa = a.b.h.data() + i * eb, *pb = d.b.dout.data() + i * eb;
        if (std::memcmp(pa, pb, eb) == 0) continue;
        if (shown < 2) {
            double va = 0, vb = 0;
            if (a.b.dtype == RAD_F32) { std::memcpy(&va, pa, 4); std::memcpy(&vb, pb, 4); }
            std::printf("  MISMATCH %-12s elem %zu: ref=%.9g dev=%.9g\n", a.name, i, va, vb);
            ++shown;
        }
        size_t c = i / cell, j = 0;
        for (; j < hist.size(); ++j) if (hist[j].first == c) break;
        if (j == hist.size()) hist.push_back({c, 0});
        hist[j].second++;
        ++bad;
    }
    std::printf("  %-12s: %zu/%zu differ across cells [", a.name, bad, a.b.n());
    for (size_t j = 0; j < hist.size() && j < 16; ++j)
        std::printf(" (%zu,%zu):%zu", hist[j].first / (cell ? sh[sh.size()-2] : 1),
                    hist[j].first % (sh[sh.size()-2] ? (size_t)sh[sh.size()-2] : 1),
                    hist[j].second);
    std::printf(" ]\n");
}

static void run_pair(Case& c, const RadKernelInfo* rk, const RadKernelInfo* dk) {
    const size_t n = c.o.size();
    std::printf("[%s] begin\n", c.title);
    std::vector<RadTensor> th(n), td(n);
    for (size_t i = 0; i < n; ++i) {
        Opd& op = c.o[i];
        op.b.snapshot();
        op.b.alloc_dev();
        if (op.present) {
            mk_tensor(th[i], op.b, op.b.h.data());
            mk_tensor(td[i], op.b, op.b.d);
        } else {
            std::memset(&th[i], 0, sizeof th[i]); std::memset(&td[i], 0, sizeof td[i]);
            th[i].rank = 1; th[i].shape[0] = 1; th[i].stride[0] = 1; th[i].dtype = RAD_F32;
            td[i] = th[i];
        }
    }
    RadArgs ah{}, ad{};
    ah.t = th.data(); ah.n_t = (int)n; ah.p = c.p.data(); ah.n_p = (int)c.p.size();
    ad.t = td.data(); ad.n_t = (int)n; ad.p = c.p.data(); ad.n_p = (int)c.p.size();

    // device run first, from the seeds; outputs start zeroed so a missed write shows up
    for (auto& op : c.o) { op.b.restore(); if (op.kind == 1) std::memset(op.b.h.data(), 0xEE, op.b.h.size()); }
    for (auto& op : c.o) op.b.to_dev();
    int rc = dk->launch(&ad, (RadStream)nullptr);
    if (rc != RAD_OK) { std::printf("%-46s DEV launch rc=%d\n", c.title, rc); ++g_fail; return; }
    CK(hipDeviceSynchronize());
    for (auto& op : c.o) if (op.kind) CK(hipMemcpy(op.b.dout.data(), op.b.d, op.b.h.size(),
                                                   hipMemcpyDeviceToHost));

    // ref run second, on freshly restored host seeds
    for (auto& op : c.o) { op.b.restore(); if (op.kind == 1) std::memset(op.b.h.data(), 0xEE, op.b.h.size()); }
    rc = rk->launch(&ah, (RadStream)nullptr);
    if (rc != RAD_OK) { std::printf("%-46s REF launch rc=%d\n", c.title, rc); ++g_fail; return; }

    int before = g_fail;
    for (int i : c.outs) compare(c.o[i], c.o[i]);
    std::printf("%-46s %s\n", c.title, g_fail == before ? "PASS (byte-exact)" : "FAIL");
}

static void params_prep(Case& c, int T, int K, int V, int width) {
    c.pi("M", T); c.pi("head_k", K); c.pi("head_v", V);
    c.pi("chunk", 64); c.pi("conv_width", width);
    c.pf("l2_eps", 1e-6); c.pf("softplus_thr", 20.0);
}

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: %s <libr3d.so> <libref.so>\n", argv[0]); return 2; }
    setvbuf(stdout, nullptr, _IONBF, 0);
    static Lib r3d, ref;
    r3d.open(argv[1]); ref.open(argv[2]);
    CK(hipSetDevice(0));

    const RadKernelInfo* rk_prep  = ref.row("gdn_conv_prep", RAD_DOMAIN_HOST);
    const RadKernelInfo* dk_prep  = r3d.row("gdn_conv_prep", RAD_DOMAIN_DEVICE);
    const RadKernelInfo* rk_fused = ref.row("gdn_conv_recurrent_update", RAD_DOMAIN_HOST);
    const RadKernelInfo* dk_fused = r3d.row("gdn_conv_recurrent_update", RAD_DOMAIN_DEVICE);
    const RadKernelInfo* rk_rec   = ref.row("gdn_recurrent_update", RAD_DOMAIN_HOST);
    const RadKernelInfo* dk_rec   = r3d.row("gdn_recurrent_update", RAD_DOMAIN_DEVICE);
    if (!rk_prep || !dk_prep || !rk_fused || !dk_fused || !rk_rec || !dk_rec) {
        std::printf("row lookup failed: prep %p/%p fused %p/%p\n",
                    (void*)rk_prep, (void*)dk_prep, (void*)rk_fused, (void*)dk_fused);
        return 2;
    }
    std::printf("rows: dev prep=%s fused=%s\n", dk_prep->name, dk_fused->name);

    std::mt19937 rng(20261021);

    // ---- conv_prep A: two sequences across the chunk boundary, cold second slot,
    //      gates fused into x, paged conv slots out of batch order.
    {
        Case c; c.title = "prep A: N2 slen70+26 K128 V64 fused-gates cold1";
        const int T = 96, K = 128, V = 64, Hg = 2, H = 6, W = 4, SL = 3, SLOTS = 4;
        const int CD = 2 * Hg * K + H * V;
        params_prep(c, T, K, V, W);
        int x   = c.add(0, "x", RAD_BF16, {T, CD + 2 * H});
        int w   = c.add(0, "w", RAD_BF16, {CD, W});
        int b   = c.add(0, "b", RAD_F32, {CD});
        int al  = c.add(0, "A_log", RAD_F32, {H});
        int db  = c.add(0, "dt_bias", RAD_F32, {H});
        int cs  = c.add(2, "conv_state", RAD_F32, {SLOTS, CD, SL});
        int cu  = c.add(0, "cu", RAD_I32, {3});
        c.o[x].present = true;
        int ia  = c.add(0, "a", RAD_F32, {1});  c.o[ia].present = false;
        int ib  = c.add(0, "b_gate", RAD_F32, {1}); c.o[ib].present = false;
        int ci  = c.add(0, "cache_idx", RAD_I32, {2});
        int hi  = c.add(0, "has_init", RAD_I32, {2});
        int q   = c.add(1, "q", RAD_BF16, {T, Hg, K});
        int k2  = c.add(1, "k", RAD_BF16, {T, Hg, K});
        int v   = c.add(1, "v", RAD_BF16, {T, H, V});
        int g   = c.add(1, "g", RAD_F32, {T, H});
        int be  = c.add(1, "beta", RAD_F32, {T, H});
        fill_bf16(rng, c.o[x].b, 1.5f); fill_bf16(rng, c.o[w].b, 1.0f);
        fill_f32(rng, c.o[b].b, 0.5f);
        fill_f32(rng, c.o[al].b, 3.0f);                    // A_log spans decay magnitudes
        fill_f32(rng, c.o[db].b, 1.0f);
        fill_f32(rng, c.o[cs].b, 2.0f);
        fill_i32({0, 70, 96}, c.o[cu].b);
        fill_i32({3, 1}, c.o[ci].b);
        fill_i32({1, 0}, c.o[hi].b);
        c.outs = {q, k2, v, g, be, cs};
        run_pair(c, rk_prep, dk_prep);
    }

    // ---- conv_prep B: one long sequence crossing two chunk resets, separate gate
    //      projections, V == K == 128, no cache_idx (slot = sequence number), warm.
    {
        Case c; c.title = "prep B: N1 slen72 K128 V128 sep-gates warm";
        const int T = 72, K = 128, V = 128, Hg = 1, H = 4, W = 4, SL = 3, SLOTS = 2;
        const int CD = 2 * Hg * K + H * V;
        params_prep(c, T, K, V, W);
        int x  = c.add(0, "x", RAD_BF16, {T, CD});
        int w  = c.add(0, "w", RAD_BF16, {CD, W});
        int b  = c.add(0, "b", RAD_F32, {CD});
        int al = c.add(0, "A_log", RAD_F32, {H});
        int db = c.add(0, "dt_bias", RAD_F32, {H});
        int cs = c.add(2, "conv_state", RAD_F32, {SLOTS, CD, SL});
        int cu = c.add(0, "cu", RAD_I32, {2});
        int ta = c.add(0, "a", RAD_BF16, {T, H});
        int tb = c.add(0, "b_gate", RAD_BF16, {T, H});
        int ci = c.add(0, "cache_idx", RAD_I32, {1}); c.o[ci].present = false;
        int hi = c.add(0, "has_init", RAD_I32, {1});  c.o[hi].present = false;
        int q  = c.add(1, "q", RAD_BF16, {T, Hg, K});
        int k2 = c.add(1, "k", RAD_BF16, {T, Hg, K});
        int v  = c.add(1, "v", RAD_BF16, {T, H, V});
        int g  = c.add(1, "g", RAD_F32, {T, H});
        int be = c.add(1, "beta", RAD_F32, {T, H});
        fill_bf16(rng, c.o[x].b, 1.5f); fill_bf16(rng, c.o[w].b, 1.0f);
        fill_f32(rng, c.o[b].b, 0.5f);
        fill_f32(rng, c.o[al].b, 3.0f); fill_f32(rng, c.o[db].b, 1.0f);
        fill_f32(rng, c.o[cs].b, 2.0f);
        fill_i32({0, 72}, c.o[cu].b);
        fill_bf16(rng, c.o[ta].b, 4.0f);                   // spans the softplus threshold
        fill_bf16(rng, c.o[tb].b, 6.0f);                   // spans sigmoid saturation
        c.outs = {q, k2, v, g, be, cs};
        run_pair(c, rk_prep, dk_prep);
    }

    // ---- fused A: two decode sequences (slen 2 each), speculative nacc {2,1} (rollback),
    //      GQA grp 3, V 64 (idle-tail threads), negative OUTPUT state_idx slot on seq1 t1.
    {
        Case c; c.title = "fused A: N2 slen2+2 nacc{2,1} K128 V64 grp3 skip";
        const int T = 4, K = 128, V = 64, Hg = 2, H = 6, W = 4, CSL = 4, CSLOTS = 4,
                  SST = 4, SW = 3, GRP = 3;      // state pool [T,...], the recorded contract
        const int CD = 2 * Hg * K + H * V;
        c.pi("q_len", 2); c.pi("head_k", K); c.pi("head_v", V); c.pi("conv_width", W);
        c.pi("n_head_v", H); c.pf("scale", 0.08838834764831845); c.pf("eps", 1e-6);
        c.pf("l2_eps", 1e-6); c.pf("softplus_thr", 20.0); c.ps("act", "silu");
        c.pi("state_fp16", 0); c.ps("state_form", "per_candidate");
        int x  = c.add(0, "x", RAD_BF16, {T, CD});
        int w  = c.add(0, "conv_w", RAD_BF16, {CD, W});
        int cb = c.add(0, "conv_b", RAD_F32, {CD});
        int cs = c.add(2, "conv_state", RAD_F32, {CSLOTS, CD, CSL});
        int ci = c.add(0, "conv_idx", RAD_I32, {2});
        int av = c.add(0, "a", RAD_BF16, {T, H});
        int bv = c.add(0, "b", RAD_BF16, {T, H});
        int al = c.add(0, "A_log", RAD_F32, {H});
        int db = c.add(0, "dt_bias", RAD_F32, {H});
        int st = c.add(2, "state", RAD_F32, {SST, H, V, K});
        int si = c.add(0, "state_idx", RAD_I32, {2, SW});
        int na = c.add(0, "num_accepted", RAD_I32, {2});
        int cu = c.add(0, "cu", RAD_I32, {3});
        int z  = c.add(0, "z", RAD_BF16, {T, H, V});
        int nw = c.add(0, "norm_w", RAD_F32, {V});
        int o  = c.add(1, "o", RAD_BF16, {T, H, V});
        int oq = c.add(1, "o_q", RAD_F32, {1});      c.o[oq].present = false;
        int os = c.add(1, "o_scale", RAD_F32, {1});  c.o[os].present = false;
        fill_bf16(rng, c.o[x].b, 1.5f); fill_bf16(rng, c.o[w].b, 1.0f);
        fill_f32(rng, c.o[cb].b, 0.5f); fill_f32(rng, c.o[cs].b, 2.0f);
        fill_bf16(rng, c.o[av].b, 4.0f); fill_bf16(rng, c.o[bv].b, 6.0f);
        fill_f32(rng, c.o[al].b, 3.0f); fill_f32(rng, c.o[db].b, 1.0f);
        fill_f32(rng, c.o[st].b, 2.0f); fill_bf16(rng, c.o[z].b, 3.0f);
        fill_f32(rng, c.o[nw].b, 2.0f);
        fill_i32({1, 3}, c.o[ci].b);
        fill_i32({0, 1, 1,  2, -1, 0}, c.o[si].b);     // seq1 in=slot 2 (disjoint); t1 write skipped
        fill_i32({2, 1}, c.o[na].b);
        fill_i32({0, 2, 4}, c.o[cu].b);
        c.outs = {o, st, cs};
        run_pair(c, rk_fused, dk_fused);
    }

    // ---- fused B: single 4-token sequence, V == K == 128 (full-width ownership), z absent
    //      (ungated norm), rolling window longer than the commit span.
    {
        Case c; c.title = "fused B: N1 slen4 K128 V128 z-absent";
        const int T = 4, K = 128, V = 128, Hg = 1, H = 3, W = 4, CSL = 6, CSLOTS = 4,
                  SST = 4, SW = 2;
        const int CD = 2 * Hg * K + H * V;
        c.pi("q_len", 4); c.pi("head_k", K); c.pi("head_v", V); c.pi("conv_width", W);
        c.pi("n_head_v", H); c.pf("scale", 0.08838834764831845); c.pf("eps", 1e-6);
        c.pf("l2_eps", 1e-6); c.pf("softplus_thr", 20.0); c.ps("act", "silu");
        c.pi("state_fp16", 0); c.ps("state_form", "per_candidate");
        int x  = c.add(0, "x", RAD_BF16, {T, CD});
        int w  = c.add(0, "conv_w", RAD_BF16, {CD, W});
        int cb = c.add(0, "conv_b", RAD_F32, {CD});
        int cs = c.add(2, "conv_state", RAD_F32, {CSLOTS, CD, CSL});
        int ci = c.add(0, "conv_idx", RAD_I32, {1});
        int av = c.add(0, "a", RAD_BF16, {T, H});
        int bv = c.add(0, "b", RAD_BF16, {T, H});
        int al = c.add(0, "A_log", RAD_F32, {H});
        int db = c.add(0, "dt_bias", RAD_F32, {H});
        int st = c.add(2, "state", RAD_F32, {SST, H, V, K});
        int si = c.add(0, "state_idx", RAD_I32, {1, SW});
        int na = c.add(0, "num_accepted", RAD_I32, {1});
        int cu = c.add(0, "cu", RAD_I32, {2});
        int z  = c.add(0, "z", RAD_BF16, {1});        c.o[z].present = false;
        int nw = c.add(0, "norm_w", RAD_F32, {V});
        int o  = c.add(1, "o", RAD_BF16, {T, H, V});
        int oq = c.add(1, "o_q", RAD_F32, {1});       c.o[oq].present = false;
        int os = c.add(1, "o_scale", RAD_F32, {1});   c.o[os].present = false;
        fill_bf16(rng, c.o[x].b, 1.5f); fill_bf16(rng, c.o[w].b, 1.0f);
        fill_f32(rng, c.o[cb].b, 0.5f); fill_f32(rng, c.o[cs].b, 2.0f);
        fill_bf16(rng, c.o[av].b, 4.0f); fill_bf16(rng, c.o[bv].b, 6.0f);
        fill_f32(rng, c.o[al].b, 3.0f); fill_f32(rng, c.o[db].b, 1.0f);
        fill_f32(rng, c.o[st].b, 2.0f); fill_f32(rng, c.o[nw].b, 2.0f);
        fill_i32({2}, c.o[ci].b);
        fill_i32({2, 1}, c.o[si].b);                  // in = si[0][0] = 2; writes to 2,1,1,1
        fill_i32({1}, c.o[na].b);
        fill_i32({0, 4}, c.o[cu].b);
        c.outs = {o, st, cs};
        run_pair(c, rk_fused, dk_fused);
    }

    // ---- fused C: fused B's geometry with the fp8 quantiser pair present (operands 16/17):
    //      ref's epilogue quantises the o row AS STORED, one e4m3 group per (token, head),
    //      scale amax/448; both planes must come back byte-identical.
    {
        Case c; c.title = "fused C: N1 slen4 K128 V128 + fp8 quant pair";
        const int T = 4, K = 128, V = 128, Hg = 1, H = 3, W = 4, CSL = 6, CSLOTS = 4,
                  SST = 4, SW = 2;
        const int CD = 2 * Hg * K + H * V;
        c.pi("q_len", 4); c.pi("head_k", K); c.pi("head_v", V); c.pi("conv_width", W);
        c.pi("n_head_v", H); c.pf("scale", 0.08838834764831845); c.pf("eps", 1e-6);
        c.pf("l2_eps", 1e-6); c.pf("softplus_thr", 20.0); c.ps("act", "silu");
        c.pi("state_fp16", 0); c.ps("state_form", "per_candidate");
        int x  = c.add(0, "x", RAD_BF16, {T, CD});
        int w  = c.add(0, "conv_w", RAD_BF16, {CD, W});
        int cb = c.add(0, "conv_b", RAD_F32, {CD});
        int cs = c.add(2, "conv_state", RAD_F32, {CSLOTS, CD, CSL});
        int ci = c.add(0, "conv_idx", RAD_I32, {1});
        int av = c.add(0, "a", RAD_BF16, {T, H});
        int bv = c.add(0, "b", RAD_BF16, {T, H});
        int al = c.add(0, "A_log", RAD_F32, {H});
        int db = c.add(0, "dt_bias", RAD_F32, {H});
        int st = c.add(2, "state", RAD_F32, {SST, H, V, K});
        int si = c.add(0, "state_idx", RAD_I32, {1, SW});
        int na = c.add(0, "num_accepted", RAD_I32, {1});
        int cu = c.add(0, "cu", RAD_I32, {2});
        int z  = c.add(0, "z", RAD_BF16, {1});        c.o[z].present = false;
        int nw = c.add(0, "norm_w", RAD_F32, {V});
        int o  = c.add(1, "o", RAD_BF16, {T, H, V});
        int oq = c.add(1, "o_q", RAD_F8E4M3, {T, H * V});
        int os = c.add(1, "o_scale", RAD_F32, {T, H});
        fill_bf16(rng, c.o[x].b, 1.5f); fill_bf16(rng, c.o[w].b, 1.0f);
        fill_f32(rng, c.o[cb].b, 0.5f); fill_f32(rng, c.o[cs].b, 2.0f);
        fill_bf16(rng, c.o[av].b, 4.0f); fill_bf16(rng, c.o[bv].b, 6.0f);
        fill_f32(rng, c.o[al].b, 3.0f); fill_f32(rng, c.o[db].b, 1.0f);
        fill_f32(rng, c.o[st].b, 2.0f); fill_f32(rng, c.o[nw].b, 2.0f);
        fill_i32({2}, c.o[ci].b);
        fill_i32({2, 1}, c.o[si].b);
        fill_i32({1}, c.o[na].b);
        fill_i32({0, 4}, c.o[cu].b);
        c.outs = {o, st, cs, oq, os};
        run_pair(c, rk_fused, dk_fused);
    }

    // ---- fused D: all-zero inputs. Every (token, head) group is then all-zero: the
    //      amax == 0 scale branch (1.0) and the all-0x00 code plane, which the 0xEE
    //      OUT prefill catches a missed write of.
    {
        Case c; c.title = "fused D: N1 slen2 K128 V128 all-zero quant groups";
        const int T = 2, K = 128, V = 128, Hg = 1, H = 2, W = 4, CSL = 6, CSLOTS = 2,
                  SST = 2, SW = 2;
        const int CD = 2 * Hg * K + H * V;
        c.pi("q_len", 2); c.pi("head_k", K); c.pi("head_v", V); c.pi("conv_width", W);
        c.pi("n_head_v", H); c.pf("scale", 0.08838834764831845); c.pf("eps", 1e-6);
        c.pf("l2_eps", 1e-6); c.pf("softplus_thr", 20.0); c.ps("act", "silu");
        c.pi("state_fp16", 0); c.ps("state_form", "per_candidate");
        int x  = c.add(0, "x", RAD_BF16, {T, CD});      // present and all-zero (Buf inits to 0)
        int w  = c.add(0, "conv_w", RAD_BF16, {CD, W});
        int cb = c.add(0, "conv_b", RAD_F32, {CD});     c.o[cb].present = false;
        int cs = c.add(2, "conv_state", RAD_F32, {CSLOTS, CD, CSL});
        int ci = c.add(0, "conv_idx", RAD_I32, {1});
        int av = c.add(0, "a", RAD_BF16, {T, H});
        int bv = c.add(0, "b", RAD_BF16, {T, H});
        int al = c.add(0, "A_log", RAD_F32, {H});
        int db = c.add(0, "dt_bias", RAD_F32, {H});
        int st = c.add(2, "state", RAD_F32, {SST, H, V, K});
        int si = c.add(0, "state_idx", RAD_I32, {1, SW});
        int na = c.add(0, "num_accepted", RAD_I32, {1});
        int cu = c.add(0, "cu", RAD_I32, {2});
        int z  = c.add(0, "z", RAD_BF16, {T, H, V});
        int nw = c.add(0, "norm_w", RAD_F32, {V});
        int o  = c.add(1, "o", RAD_BF16, {T, H, V});
        int oq = c.add(1, "o_q", RAD_F8E4M3, {T, H * V});
        int os = c.add(1, "o_scale", RAD_F32, {T, H});
        fill_i32({0}, c.o[ci].b);
        fill_i32({0, 0}, c.o[si].b);
        fill_i32({1}, c.o[na].b);
        fill_i32({0, 2}, c.o[cu].b);
        c.outs = {o, st, cs, oq, os};
        run_pair(c, rk_fused, dk_fused);
    }

    // ---- rec A: the NON-fused gdn_recurrent_update row with the pair — the registry lists
    //      o_q/o_scale on both recurrent ops, so both kernels must carry the epilogue.
    {
        Case c; c.title = "rec A: N2 slen2+1 nacc{2,1} K128 V128 + fp8 quant pair";
        const int T = 3, K = 128, V = 128, Hg = 2, H = 4, SST = 4, SW = 3;
        c.pi("q_len", 2); c.pi("head_k", K); c.pi("head_v", V); c.pi("n_head_v", H);
        c.pf("scale", 0.08838834764831845); c.pf("eps", 1e-6);
        c.pf("l2_eps", 1e-6); c.pf("softplus_thr", 20.0); c.ps("act", "silu");
        c.pi("state_fp16", 0); c.ps("state_form", "per_candidate");
        int q  = c.add(0, "q", RAD_BF16, {T, Hg, K});
        int k2 = c.add(0, "k", RAD_BF16, {T, Hg, K});
        int v  = c.add(0, "v", RAD_BF16, {T, H, V});
        int av = c.add(0, "a", RAD_BF16, {T, H});
        int bv = c.add(0, "b", RAD_BF16, {T, H});
        int al = c.add(0, "A_log", RAD_F32, {H});
        int db = c.add(0, "dt_bias", RAD_F32, {H});
        int st = c.add(2, "state", RAD_F32, {SST, H, V, K});
        int si = c.add(0, "state_idx", RAD_I32, {2, SW});
        int na = c.add(0, "num_accepted", RAD_I32, {2});
        int cu = c.add(0, "cu", RAD_I32, {3});
        int z  = c.add(0, "z", RAD_BF16, {T, H, V});
        int nw = c.add(0, "norm_w", RAD_F32, {V});
        int o  = c.add(1, "o", RAD_BF16, {T, H, V});
        int oq = c.add(1, "o_q", RAD_F8E4M3, {T, H * V});
        int os = c.add(1, "o_scale", RAD_F32, {T, H});
        fill_bf16(rng, c.o[q].b, 1.2f); fill_bf16(rng, c.o[k2].b, 1.2f);
        fill_bf16(rng, c.o[v].b, 1.2f);
        fill_bf16(rng, c.o[av].b, 4.0f); fill_bf16(rng, c.o[bv].b, 6.0f);
        fill_f32(rng, c.o[al].b, 3.0f); fill_f32(rng, c.o[db].b, 1.0f);
        fill_f32(rng, c.o[st].b, 2.0f); fill_bf16(rng, c.o[z].b, 3.0f);
        fill_f32(rng, c.o[nw].b, 2.0f);
        fill_i32({0, 1, 1,  2, 2, 2}, c.o[si].b);   // seq0 in=1 writes {0,1}; seq1 in=2 writes {2}
        fill_i32({2, 1}, c.o[na].b);
        fill_i32({0, 2, 3}, c.o[cu].b);
        c.outs = {o, st, oq, os};
        run_pair(c, rk_rec, dk_rec);
    }

    std::printf("\n%s: %d mismatch group(s)\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
