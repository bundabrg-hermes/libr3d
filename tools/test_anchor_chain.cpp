/* test_anchor_chain.cpp -- CHAINED differential test for the anchor (speculative) state
 * form of the two recurrent GDN rows, device (libr3d) vs the libref oracle, byte for byte.
 *
 * Why this exists when kbench passes: every kbench case is one launch on freshly seeded
 * operands. The anchor form's contract is CHAINED -- step t writes factor copies and the
 * shift stamps that step t+1's fold replays, through a state_idx window the caller slides
 * between steps. A seed cannot witness that handoff; only a chain can. This tool runs the
 * same step plan against the same state buffer on both sides -- ref's committed output is
 * the next step's seed on both sides -- over window slides, duplicate factor blocks
 * (dup_f), the stamp walk (a stable column reading back its own stamp), rollbacks
 * (num_accepted), a fold-only sequence (zero rows), a padded row (negative column), and a
 * na==0 step. Any divergence prints the cell histogram and fails.
 *
 * Build (in radiance-dev, on the card):
 *   hipcc -O2 --offload-arch=gfx1100 -I/opt/radiance/include/radiance \
 *       tools/test_anchor_chain.cpp -ldl -o /tmp/test_anchor_chain
 * Run:
 *   /tmp/test_anchor_chain <libr3d.so> <libref.so>
 */
#include <hip/hip_runtime.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
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

static int dt_bytes(uint32_t d) {
    if (d == RAD_F8E4M3 || d == RAD_F8E5M2 || d == RAD_I8 || RAD_U8 == d) return 1;
    return (d == RAD_F32 || d == RAD_I32) ? 4 : 2;
}

struct Buf {
    std::vector<uint8_t> h, seed, dout;
    void* d = nullptr; size_t cap = 0;
    uint32_t dtype = 0;
    std::vector<int64_t> shape;
    size_t n() const { size_t r = 1; for (auto s : shape) r *= (size_t)s; return r; }
    void init(uint32_t dt, std::vector<int64_t> sh) {
        dtype = dt; shape = std::move(sh);
        size_t bytes = n() * dt_bytes(dt); if (!bytes) bytes = 16;
        h.assign(bytes, 0); seed.assign(bytes, 0); dout.assign(bytes, 0);
    }
    void set_rows(int64_t r) {                       // live ragged batches size to cu's tail
        shape[0] = r;
        size_t bytes = n() * dt_bytes(dtype);
        h.resize(bytes); seed.resize(bytes); dout.resize(bytes);
    }
    void snapshot() { seed = h; }
    void restore()  { h = seed; }
    void alloc_dev() { if (cap < h.size()) { if (d) hipFree(d); CK(hipMalloc(&d, h.size())); cap = h.size(); } }
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
        p[i] = (uint16_t)(u32 >> 16);
    }
}
static void set_i32(Buf& b, std::initializer_list<int32_t> v) {
    size_t i = 0;
    for (int32_t x : v) { if (i >= b.n()) break; ((int32_t*)b.h.data())[i++] = x; }
}

static int g_fail = 0;

struct Opd { Buf b; int kind = 0; const char* name = ""; bool present = true; };

struct ChainCase {
    std::vector<Opd> o;
    std::vector<RadParam> p;
    std::vector<int> outs;                       // compared EVERY step (state included)

    int add(int kind, const char* name, uint32_t dt, std::vector<int64_t> sh) {
        o.push_back(Opd{}); o.back().kind = kind; o.back().name = name;
        o.back().b.init(dt, std::move(sh));
        return (int)o.size() - 1;
    }
    void pi(const char* k, long long v) { p.push_back(RadParam{k, RAD_P_INT, v, 0, nullptr, 0}); }
    void pf(const char* k, double v)    { p.push_back(RadParam{k, RAD_P_F64, 0, 0, nullptr, v}); }
    void ps(const char* k, const char* v) { p.push_back(RadParam{k, RAD_P_STR, 0, 0, v, 0}); }
};

static void compare(const Opd& a) {
    if (std::memcmp(a.b.h.data(), a.b.dout.data(), a.b.h.size()) == 0) return;
    ++g_fail;
    size_t eb = dt_bytes(a.b.dtype), bad = 0, shown = 0;
    const auto& sh = a.b.shape;
    for (size_t i = 0; i < a.b.n(); ++i) {
        const uint8_t* pa = a.b.h.data() + i * eb, *pb = a.b.dout.data() + i * eb;
        if (std::memcmp(pa, pb, eb) == 0) continue;
        if (shown < 6) {
            double va = 0, vb = 0;
            if (a.b.dtype == RAD_F32) { std::memcpy(&va, pa, 4); std::memcpy(&vb, pb, 4); }
            else if (a.b.dtype == RAD_BF16) {
                uint32_t ua = ((uint32_t)*(const uint16_t*)pa) << 16, ub = ((uint32_t)*(const uint16_t*)pb) << 16;
                float fa, fb; std::memcpy(&fa, &ua, 4); std::memcpy(&fb, &ub, 4); va = fa; vb = fb;
            }
            // decompose the element index for [rows, heads, V]-shaped outputs
            std::printf("  MISMATCH %-12s elem %zu (row %zu head %zu v %zu): ref=%.9g dev=%.9g\n",
                        a.name, i, sh.size() >= 3 ? i / (size_t)(sh[1] * sh[2]) : 0,
                        sh.size() >= 3 ? (i / (size_t)sh[2]) % (size_t)sh[1] : 0,
                        sh.size() >= 3 ? i % (size_t)sh[2] : 0, va, vb);
            ++shown;
        }
        ++bad;
    }
    std::printf("  %-12s: %zu/%zu differ\n", a.name, bad, a.b.n());
}

/* One chained step: inputs re-filled by the caller, then device and ref on identical seeds.
 * The INOUT state keeps whatever the ref committed (which equals the device output iff this
 * step passed) -- that becomes the next step's seed on both sides. */
static void run_step(ChainCase& c, const RadKernelInfo* rk, const RadKernelInfo* dk,
                     const char* title) {
    const size_t n = c.o.size();
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

    for (auto& op : c.o) { op.b.restore(); if (op.kind == 1) std::memset(op.b.h.data(), 0xEE, op.b.h.size()); }
    for (auto& op : c.o) op.b.to_dev();
    int rc = dk->launch(&ad, (RadStream)nullptr);
    if (rc != RAD_OK) { std::printf("%-52s DEV launch rc=%d\n", title, rc); ++g_fail; return; }
    CK(hipDeviceSynchronize());
    for (auto& op : c.o) if (op.kind) CK(hipMemcpy(op.b.dout.data(), op.b.d, op.b.h.size(),
                                                   hipMemcpyDeviceToHost));

    for (auto& op : c.o) { op.b.restore(); if (op.kind == 1) std::memset(op.b.h.data(), 0xEE, op.b.h.size()); }
    rc = rk->launch(&ah, (RadStream)nullptr);
    if (rc != RAD_OK) { std::printf("%-52s REF launch rc=%d\n", title, rc); ++g_fail; return; }

    int before = g_fail;
    for (int i : c.outs) compare(c.o[i]);
    std::printf("%-52s %s\n", title, g_fail == before ? "PASS (byte-exact)" : "FAIL");
}

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: %s <libr3d.so> <libref.so>\n", argv[0]); return 2; }
    setvbuf(stdout, nullptr, _IONBF, 0);
    static Lib r3d, ref;
    r3d.open(argv[1]); ref.open(argv[2]);
    CK(hipSetDevice(0));

    const RadKernelInfo* rk_rec   = ref.row("gdn_recurrent_update", RAD_DOMAIN_HOST);
    const RadKernelInfo* dk_rec   = r3d.row("gdn_recurrent_update", RAD_DOMAIN_DEVICE);
    const RadKernelInfo* rk_fused = ref.row("gdn_conv_recurrent_update", RAD_DOMAIN_HOST);
    const RadKernelInfo* dk_fused = r3d.row("gdn_conv_recurrent_update", RAD_DOMAIN_DEVICE);
    if (!rk_rec || !dk_rec || !rk_fused || !dk_fused) {
        std::printf("row lookup failed\n"); return 2;
    }
    std::printf("rows: %s / %s\n", dk_rec->name, dk_fused->name);

    std::mt19937 rng(20261021);
    const int K = 128, V = 128, Hv = 4, Hg = 1, GRP = Hv / Hg, SLOTS = 8, N = 2, SW = 6;
    // conv_state_len must hold width-2 + a full spec-3 commit: 2 + 4 = 6 (ref enforces this)
    const int CD = 2 * Hg * K + Hv * V, CSLOTS = 4, CSL = 6;

    /* ---------- chain 1: the non-fused recurrent row, four+ steps ---------- */
    {
        ChainCase c;
        c.pi("head_k", K); c.pi("head_v", V); c.pi("n_head_v", Hv);
        c.pf("scale", 0.08838834764831845); c.pf("eps", 1e-6); c.pf("l2_eps", 1e-6);
        c.pf("softplus_thr", 20.0); c.ps("act", "silu"); c.pi("state_fp16", 0);
        c.ps("state_form", "anchor");
        int q  = c.add(0, "q", RAD_BF16, {8, Hg, K});
        int k2 = c.add(0, "k", RAD_BF16, {8, Hg, K});
        int v  = c.add(0, "v", RAD_BF16, {8, Hv, V});
        int av = c.add(0, "a", RAD_BF16, {8, Hv});
        int bv = c.add(0, "b", RAD_BF16, {8, Hv});
        int al = c.add(0, "A_log", RAD_F32, {Hv});
        int db = c.add(0, "dt_bias", RAD_F32, {Hv});
        int st = c.add(2, "state", RAD_F32, {SLOTS, Hv, V, K});
        int si = c.add(0, "state_idx", RAD_I32, {N, SW});
        int na = c.add(0, "num_accepted", RAD_I32, {N});
        int cu = c.add(0, "cu", RAD_I32, {N + 1});
        int z  = c.add(0, "z", RAD_BF16, {8, Hv, V});
        int nw = c.add(0, "norm_w", RAD_F32, {V});
        int o  = c.add(1, "o", RAD_BF16, {8, Hv, V});
        int oq = c.add(1, "o_q", RAD_F32, {1});      c.o[oq].present = false;
        int os = c.add(1, "o_scale", RAD_F32, {1});  c.o[os].present = false;
        c.outs = {st, o};
        // Live boot hands the pool ZEROED slots: the boot fold replays an identity factor
        // block (k=0, beta=0, stamp 0 -> column 2). Random state would replay NaN-yield
        // garbage no engine ever feeds -- a NaN-sign-only domain, not the contract.
        std::memset(c.o[st].b.h.data(), 0, c.o[st].b.h.size());
        fill_f32(rng, c.o[al].b, 3.0f); fill_f32(rng, c.o[db].b, 1.0f);
        fill_f32(rng, c.o[nw].b, 1.0f);
        c.p.push_back(RadParam{"q_len", RAD_P_INT, 4, 0, nullptr, 0});   // declared bound

        // step 0: boot from a random state, seq1 FOLD-ONLY (zero rows: tensors sized to
        // cu's tail), nacc 1
        for (int i : {q, k2, v, av, bv, z, o}) c.o[i].b.set_rows(4);
        fill_bf16(rng, c.o[q].b, 1.0f); fill_bf16(rng, c.o[k2].b, 1.0f);
        fill_bf16(rng, c.o[v].b, 2.0f); fill_bf16(rng, c.o[av].b, 4.0f);
        fill_bf16(rng, c.o[bv].b, 6.0f); fill_bf16(rng, c.o[z].b, 2.0f);
        set_i32(c.o[si].b, {2,2,2,5,5,5,  0,0,0,3,3,3});
        set_i32(c.o[na].b, {1, 1});
        set_i32(c.o[cu].b, {0, 4, 4});
        run_step(c, rk_rec, dk_rec, "rec-anchor 0: boot, fold-only seq1, random stamps");

        // step 1: window slides (col3 != col4 -> dup_f), 3 accepted on seq0, 0 on seq1.
        // Slots are DISJOINT across sequences — the real slot manager never aliases them,
        // and block-parallel writers cannot share one the way the host's sequential loop can.
        for (int i : {q, k2, v, av, bv, z, o}) c.o[i].b.set_rows(8);
        fill_bf16(rng, c.o[q].b, 1.0f); fill_bf16(rng, c.o[k2].b, 1.0f);
        fill_bf16(rng, c.o[v].b, 2.0f); fill_bf16(rng, c.o[av].b, 4.0f);
        fill_bf16(rng, c.o[bv].b, 6.0f); fill_bf16(rng, c.o[z].b, 2.0f);
        set_i32(c.o[si].b, {3,4,4,6,7,7,  0,1,1,2,2,2});
        set_i32(c.o[na].b, {3, 0});
        set_i32(c.o[cu].b, {0, 4, 8});
        run_step(c, rk_rec, dk_rec, "rec-anchor 1: dup_f slide, accept 3 then 0");

        // step 2: stable window (stamp walk: col4 block carries this column's stamp from
        // step 1's publish), rollback-shaped nacc, seq1 row PADDED (negative factor col)
        fill_bf16(rng, c.o[q].b, 1.0f); fill_bf16(rng, c.o[k2].b, 1.0f);
        fill_bf16(rng, c.o[v].b, 2.0f); fill_bf16(rng, c.o[av].b, 4.0f);
        fill_bf16(rng, c.o[bv].b, 6.0f); fill_bf16(rng, c.o[z].b, 2.0f);
        set_i32(c.o[si].b, {3,4,4,6,7,7,  1,1,1,2,-1,2});
        set_i32(c.o[na].b, {4, 2});
        set_i32(c.o[cu].b, {0, 4, 8});
        run_step(c, rk_rec, dk_rec, "rec-anchor 2: stamp walk, full rollback, padded row");

        // step 3: na==0 everywhere (no fold), anchor must still advance by its own tokens
        fill_bf16(rng, c.o[q].b, 1.0f); fill_bf16(rng, c.o[k2].b, 1.0f);
        fill_bf16(rng, c.o[v].b, 2.0f); fill_bf16(rng, c.o[av].b, 4.0f);
        fill_bf16(rng, c.o[bv].b, 6.0f); fill_bf16(rng, c.o[z].b, 2.0f);
        set_i32(c.o[si].b, {4,4,4,4,4,4,  0,0,0,1,1,1});
        set_i32(c.o[na].b, {0, 0});
        set_i32(c.o[cu].b, {0, 2, 8});
        run_step(c, rk_rec, dk_rec, "rec-anchor 3: na 0, ragged slen 2+6");
    }

    /* ---------- chain 2: the fused conv+recurrent row (the live spec3 band) ---------- */
    {
        ChainCase c;
        c.pi("head_k", K); c.pi("head_v", V); c.pi("n_head_v", Hv); c.pi("conv_width", 4);
        c.pf("scale", 0.08838834764831845); c.pf("eps", 1e-6); c.pf("l2_eps", 1e-6);
        c.pf("softplus_thr", 20.0); c.ps("act", "silu"); c.pi("state_fp16", 0);
        c.ps("state_form", "anchor");
        int x  = c.add(0, "x", RAD_BF16, {8, CD});
        int w  = c.add(0, "conv_w", RAD_BF16, {CD, 4});
        int cb = c.add(0, "conv_b", RAD_F32, {CD});
        int cs = c.add(2, "conv_state", RAD_F32, {CSLOTS, CD, CSL});
        int ci = c.add(0, "conv_idx", RAD_I32, {N});
        int av = c.add(0, "a", RAD_BF16, {8, Hv});
        int bv = c.add(0, "b", RAD_BF16, {8, Hv});
        int al = c.add(0, "A_log", RAD_F32, {Hv});
        int db = c.add(0, "dt_bias", RAD_F32, {Hv});
        int st = c.add(2, "state", RAD_F32, {SLOTS, Hv, V, K});
        int si = c.add(0, "state_idx", RAD_I32, {N, SW});
        int na = c.add(0, "num_accepted", RAD_I32, {N});
        int cu = c.add(0, "cu", RAD_I32, {N + 1});
        int z  = c.add(0, "z", RAD_BF16, {8, Hv, V});
        int nw = c.add(0, "norm_w", RAD_F32, {V});
        int o  = c.add(1, "o", RAD_BF16, {8, Hv, V});
        int oq = c.add(1, "o_q", RAD_F8E4M3, {8, Hv, V});
        int os = c.add(1, "o_scale", RAD_F32, {8, Hv});
        c.outs = {st, o, cs, oq, os};
        std::memset(c.o[st].b.h.data(), 0, c.o[st].b.h.size());   // zeroed pool, as live
        fill_f32(rng, c.o[cs].b, 2.0f);
        fill_bf16(rng, c.o[w].b, 1.0f); fill_f32(rng, c.o[cb].b, 0.5f);
        fill_f32(rng, c.o[al].b, 3.0f); fill_f32(rng, c.o[db].b, 1.0f);
        fill_f32(rng, c.o[nw].b, 1.0f);
        c.p.push_back(RadParam{"q_len", RAD_P_INT, 4, 0, nullptr, 0});

        // step 0: boot; seq1 fold-only (the conv leg must NOT commit its shift); nacc 1
        for (int i : {x, av, bv, z, o, oq, os}) c.o[i].b.set_rows(4);
        fill_bf16(rng, c.o[x].b, 1.5f); fill_bf16(rng, c.o[av].b, 4.0f);
        fill_bf16(rng, c.o[bv].b, 6.0f); fill_bf16(rng, c.o[z].b, 2.0f);
        set_i32(c.o[ci].b, {0, 2});
        set_i32(c.o[si].b, {2,2,2,5,5,5,  0,0,0,3,3,3});
        set_i32(c.o[na].b, {1, 1});
        set_i32(c.o[cu].b, {0, 4, 4});
        run_step(c, rk_fused, dk_fused, "fused-anchor 0: boot, fold-only seq1 conv skip");

        // step 1: slide + dup_f, accept 3 then 1 (the conv window reads at na-1; ref's
        // conv leg refuses na==0 outright, and the engine never issues it)
        for (int i : {x, av, bv, z, o, oq, os}) c.o[i].b.set_rows(8);
        fill_bf16(rng, c.o[x].b, 1.5f); fill_bf16(rng, c.o[av].b, 4.0f);
        fill_bf16(rng, c.o[bv].b, 6.0f); fill_bf16(rng, c.o[z].b, 2.0f);
        set_i32(c.o[ci].b, {0, 2});
        set_i32(c.o[si].b, {3,4,4,6,7,7,  0,1,1,2,2,2});
        set_i32(c.o[na].b, {3, 1});
        set_i32(c.o[cu].b, {0, 4, 8});
        run_step(c, rk_fused, dk_fused, "fused-anchor 1: slide, accept 3/1, window shift");

        // step 2: stamp walk + rollback + padded row
        fill_bf16(rng, c.o[x].b, 1.5f); fill_bf16(rng, c.o[av].b, 4.0f);
        fill_bf16(rng, c.o[bv].b, 6.0f); fill_bf16(rng, c.o[z].b, 2.0f);
        set_i32(c.o[ci].b, {1, 3});
        set_i32(c.o[si].b, {3,4,4,6,7,7,  1,1,1,2,-1,2});
        set_i32(c.o[na].b, {4, 2});
        set_i32(c.o[cu].b, {0, 4, 8});
        run_step(c, rk_fused, dk_fused, "fused-anchor 2: stamp walk, rollback, padded row");

        // step 3: na==0 everywhere, ragged slen 2+4 (the conv cap: width-2+4 == CSL),
        // anchor columns stable (stamp walk #2)
        for (int i : {x, av, bv, z, o, oq, os}) c.o[i].b.set_rows(6);
        fill_bf16(rng, c.o[x].b, 1.5f); fill_bf16(rng, c.o[av].b, 4.0f);
        fill_bf16(rng, c.o[bv].b, 6.0f); fill_bf16(rng, c.o[z].b, 2.0f);
        set_i32(c.o[ci].b, {0, 2});
        set_i32(c.o[si].b, {4,4,4,4,4,4,  0,0,0,1,1,1});
        set_i32(c.o[na].b, {1, 1});
        set_i32(c.o[cu].b, {0, 2, 6});
        run_step(c, rk_fused, dk_fused, "fused-anchor 3: ragged slen 2+4, fold 1, stamps");
    }

    std::printf("\n%s (%d failures)\n", g_fail ? "FAILURES" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
