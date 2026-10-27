/* test_qk_fuse.cpp -- differential byte-identity test for the Q27 fused prologue rows.
 *
 * Why a unit test and not kbench: qk_norm_rope_gate is not docs/OPS.md vocabulary, so the
 * packs carry no records and libref has no shapes for it -- kbench skips it by name. The
 * claim under test is exactly the template's substitution claim: qk_norm_rope_gate is
 * BYTE-IDENTICAL to rmsnorm(q) + rmsnorm(k) + rope(q) + rope(k), and the kv_store fold adds
 * precisely what kv_store_bf16 writes. Both sides therefore run the SHIPPED libr3d rows --
 * the references are the real chain kernels, not a reimplementation. The cos/sin table the
 * fused side reads is filled by rope_table_f32/bf16 itself, so the test also proves the
 * table/rope ladder agreement the fusion contract rests on (r3d_rope.hip's rope_table header).
 *
 * Padded positions (negative) are deliberately NOT byte-compared on q/k: the chain's rope
 * rotates them with the angle of position -1 while the fused kernel clamps to table row 0 --
 * both are defined garbage for rows the engine never reads. They ARE used to prove the cache
 * rule (a negative or out-of-range slot stores nothing).
 *
 * Build (in radiance-dev, on the card):
 *   hipcc -O2 --offload-arch=gfx1100 -I/opt/radiance/include/radiance \
 *       tools/test_qk_fuse.cpp -ldl -o /tmp/test_qk_fuse
 * Run:
 *   /tmp/test_qk_fuse <libr3d.so>
 */
#include <hip/hip_runtime.h>
#include <dlfcn.h>

#include <algorithm>
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

static int g_fail = 0, g_pass = 0;

struct Lib {
    void* h = nullptr; CountFn count = nullptr; AtFn at = nullptr;
    void open(const char* path) {
        h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!h) { std::printf("dlopen %s: %s\n", path, dlerror()); std::exit(1); }
        count = (CountFn)dlsym(h, "rad_kernel_count");
        at    = (AtFn)dlsym(h, "rad_kernel_at");
        if (!count || !at) { std::printf("dlsym failed in %s\n", path); std::exit(1); }
    }
    const RadKernelInfo* row(const char* op) {
        for (int i = 0; i < count(); ++i) {
            const RadKernelInfo* k = at(i);
            if (k && std::strcmp(k->op, op) == 0 && k->domain == RAD_DOMAIN_DEVICE) return k;
        }
        return nullptr;
    }
};

static int dt_bytes(uint32_t d) {
    if (d == RAD_F8E4M3 || d == RAD_F8E5M2 || d == RAD_I8 || d == RAD_U8) return 1;
    if (d == RAD_I64) return 8;
    return (d == RAD_F32 || d == RAD_I32) ? 4 : 2;
}

/* A device-backed operand: one host image `h` (source of truth), one device copy `d`. The
 * host image is pushed before every launch and pulled after, so multi-stage chains compose
 * in host state exactly as the engine composes them in device memory. A custom view (shape,
 * stride) may alias another buffer's device image (the rmsnorm strided-slice views). */
struct Buf {
    std::vector<uint8_t> h;
    void* d = nullptr;
    uint32_t dtype = 0;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;              // custom, or natural (empty = row-major)
    bool present = true;
    Buf* alias = nullptr;                      // device image + host image come from this
    size_t n() const { size_t r = 1; for (auto s : shape) r *= (size_t)s; return r; }
    std::vector<uint8_t>* host() { return alias ? &alias->h : &h; }
    void init(uint32_t dt, std::vector<int64_t> sh) {
        dtype = dt; shape = std::move(sh);
        size_t bytes = n() * dt_bytes(dt); if (!bytes) bytes = 16;
        h.assign(bytes, 0);
    }
    void alloc_dev() {
        if (alias) { alias->alloc_dev(); return; }
        if (!d) CK(hipMalloc(&d, h.size()));
    }
    void* dev() { return alias ? alias->d : d; }
    void mk(RadTensor& t) {
        if (!present) {
            std::memset(&t, 0, sizeof t);
            t.rank = 1; t.shape[0] = 1; t.stride[0] = 1; t.dtype = RAD_F32; t.data = nullptr;
            return;
        }
        t.data = dev(); t.dtype = dtype; t.rank = (uint32_t)shape.size();
        if (strides.size() == shape.size()) {
            for (size_t i = 0; i < shape.size(); ++i) { t.shape[i] = shape[i]; t.stride[i] = strides[i]; }
        } else {
            int64_t s = 1;
            for (int i = (int)shape.size() - 1; i >= 0; --i) {
                t.shape[i] = shape[i]; t.stride[i] = s; s *= shape[i];
            }
        }
    }
    ~Buf() { if (d) { hipError_t e = hipFree(d); if (e) std::fprintf(stderr,
        "hipFree FAIL %s\n", hipGetErrorString(e)); } }
};

struct Param { RadParam p; };
static RadParam Pi(const char* k, long long v) { return RadParam{k, RAD_P_INT, v, 0, nullptr, 0}; }
static RadParam Pf(const char* k, double v)    { return RadParam{k, RAD_P_F64, 0, 0, nullptr, v}; }
static RadParam Ps(const char* k, const char* v) { return RadParam{k, RAD_P_STR, 0, 0, v, 0}; }

/* Launch `row` with the operands/params; push every operand first, pull every operand after. */
static int run(const RadKernelInfo* row, std::vector<Buf*>& ops, std::vector<RadParam>& ps) {
    std::vector<RadTensor> ts(ops.size());
    for (size_t i = 0; i < ops.size(); ++i) { ops[i]->alloc_dev(); ops[i]->mk(ts[i]); }
    if (ops.size()) {
        for (auto* o : ops) if (o->present && o->host()->size())
            CK(hipMemcpy(o->dev(), o->host()->data(), o->host()->size(), hipMemcpyHostToDevice));
    }
    RadArgs a{}; a.t = ts.data(); a.n_t = (int)ts.size(); a.p = ps.data(); a.n_p = (int)ps.size();
    int rc = row->launch(&a, (RadStream)nullptr);
    CK(hipDeviceSynchronize());
    for (auto* o : ops) if (o->present && o->host()->size())
        CK(hipMemcpy(o->host()->data(), o->dev(), o->host()->size(), hipMemcpyDeviceToHost));
    return rc;
}

static void cmp(const char* what, const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    if (a == b) { ++g_pass; return; }
    ++g_fail;
    size_t shown = 0, bad = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) if (a[i] != b[i]) {
        if (shown < 3) { std::printf("  MISMATCH %s byte %zu: chain=%02x fused=%02x\n",
                                     what, i, a[i], b[i]); ++shown; }
        ++bad;
    }
    std::printf("  %s: %zu/%zu bytes differ\n", what, bad, a.size());
}

struct Geo { int hd, rot, M, nq, nkv, bs, cs_f32, w_f32, pos_i64; };

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

/* One differential case: run the chain pipeline and the fused row on identical seeds and
 * compare every output byte for byte. `fuse_kv` selects the fold row; `gated` materializes
 * gate_out; `pad` makes row 0 a padded token (slot/position negative). */
static void case_run(Lib& lib, std::mt19937& rng, const Geo& g, bool fuse_kv,
                     bool gated, bool pad) {
    const int CS_ROWS = 64, nb = (g.M + g.bs - 1) / g.bs + 2;
    char tag[160];
    std::snprintf(tag, sizeof tag, "%s hd=%d rot=%d M=%d nq=%d nkv=%d cs=%s w=%s pos=%s%s%s",
                  fuse_kv ? "KV" : "plain", g.hd, g.rot, g.M, g.nq, g.nkv,
                  g.cs_f32 ? "f32" : "bf16", g.w_f32 ? "f32" : "bf16",
                  g.pos_i64 ? "i64" : "i32", gated ? " gate" : "", pad ? " pad" : "");

    std::vector<Buf*> keep;
    auto mkbuf = [&](uint32_t dt, std::vector<int64_t> sh) {
        Buf* b = new Buf(); b->init(dt, std::move(sh)); keep.push_back(b); return b;
    };

    Buf* qg  = mkbuf(RAD_BF16, {g.M, (long long)g.nq * 2 * g.hd});
    Buf* k_c = mkbuf(RAD_BF16, {g.M, (long long)g.nkv * g.hd});
    Buf* k_f = mkbuf(RAD_BF16, {g.M, (long long)g.nkv * g.hd});
    Buf* v   = mkbuf(RAD_BF16, {g.M, (long long)g.nkv * g.hd});
    Buf* wq  = mkbuf(g.w_f32 ? RAD_F32 : RAD_BF16, {g.hd});
    Buf* wk  = mkbuf(g.w_f32 ? RAD_F32 : RAD_BF16, {g.hd});
    Buf* pos = mkbuf(g.pos_i64 ? RAD_I64 : RAD_I32, {g.M});
    Buf* slot = mkbuf(RAD_I32, {g.M});
    Buf* cs  = mkbuf(g.cs_f32 ? RAD_F32 : RAD_BF16, {CS_ROWS, g.rot});
    Buf* qc  = mkbuf(RAD_BF16, {g.M, (long long)g.nq * g.hd});
    Buf* qf  = mkbuf(RAD_BF16, {g.M, (long long)g.nq * g.hd});
    Buf* cc  = mkbuf(RAD_BF16, {nb, g.nkv, g.bs, 2 * g.hd});
    Buf* cf  = mkbuf(RAD_BF16, {nb, g.nkv, g.bs, 2 * g.hd});
    Buf* gof = mkbuf(RAD_BF16, {g.M, (long long)g.nq * g.hd});

    fill_bf16(rng, *qg, 1.0f); fill_bf16(rng, *k_c, 1.0f); fill_bf16(rng, *v, 1.0f);
    if (g.w_f32) fill_f32(rng, *wq, 0.5f), fill_f32(rng, *wk, 0.5f);
    else         fill_bf16(rng, *wq, 0.5f), fill_bf16(rng, *wk, 0.5f);
    for (int i = 0; i < g.M; ++i) {
        int p = (int)(rng() % (unsigned)(CS_ROWS - 1)) + 1;
        if (pad && i == 0) p = -1;
        if (g.pos_i64) ((int64_t*)pos->h.data())[i] = p; else ((int32_t*)pos->h.data())[i] = p;
        int32_t s = i * 3 % (nb * g.bs);                 // dense-ish, never >= nb*bs
        if (pad && i == 0) s = -1;
        ((int32_t*)slot->h.data())[i] = s;
    }

    std::vector<uint8_t> k_seed = k_c->h;
    const double theta = 1000000.0, eps = 1e-6, wadd = 1.0;
    if (getenv("QK_DIAG")) std::fprintf(stderr, "PHASE start %s\n", tag);

    auto rms_row = lib.row("rmsnorm"), rope_row = lib.row("rope");
    auto rt_row = lib.row("rope_table"), ks_row = lib.row("kv_store");
    auto fused_row = lib.row(fuse_kv ? "qk_norm_rope_gate_kv_store" : "qk_norm_rope_gate");
    if (!rms_row || !rope_row || !rt_row || !ks_row || !fused_row) {
        std::printf("%-52s MISSING ROW\n", tag); ++g_fail;
        for (auto* b : keep) delete b; return;
    }

    /* ---- chain: rope_table -> rmsnorm(q) -> rmsnorm(k) -> rope(q) -> rope(k) -> kv_store */
    std::memset(qc->h.data(), 0, qc->h.size());
    std::memset(cc->h.data(), 0xEE, cc->h.size());
    {   /* rope_table fills the plane the fused side reads, and the table row for the padded
         * position is written nowhere -- exactly the row the fused kernel must not read. */
        std::vector<Buf*> o{pos, cs};
        std::vector<RadParam> p{Pi("M", g.M), Pi("rot", g.rot), Pf("theta", theta),
                                Pf("scale", 1.0), Ps("dtype", g.cs_f32 ? "f32" : "bf16")};
        int rc = run(rt_row, o, p);
        if (rc) { std::printf("%-52s rope_table rc=%d\n", tag, rc); ++g_fail; }
    }
    if (g.nq > 0) {   /* rmsnorm over the strided q slice of [q|gate]: last extent hd, pitch 2*hd */
        Buf qv; qv.dtype = RAD_BF16; qv.shape = {g.M * g.nq, g.hd};
        qv.strides = {(long long)2 * g.hd, 1}; qv.alias = qg;
        std::vector<Buf*> o{&qv, wq, qc};
        std::vector<RadParam> p{Pi("M", (long long)g.M * g.nq), Pi("n", g.hd), Pf("eps", eps),
                                Pf("wadd", wadd), Ps("dtype", "bf16")};
        int rc = run(rms_row, o, p);
        if (rc) { std::printf("%-52s rmsnorm(q) rc=%d\n", tag, rc); ++g_fail; }
    }
    {   /* rmsnorm over k, in place: y aliases x */
        Buf kv; kv.dtype = RAD_BF16; kv.shape = {g.M * g.nkv, g.hd}; kv.alias = k_c;
        Buf ky; ky.dtype = RAD_BF16; ky.shape = {g.M * g.nkv, g.hd}; ky.alias = k_c;
        std::vector<Buf*> o{&kv, wk, &ky};
        std::vector<RadParam> p{Pi("M", (long long)g.M * g.nkv), Pi("n", g.hd), Pf("eps", eps),
                                Pf("wadd", wadd), Ps("dtype", "bf16")};
        int rc = run(rms_row, o, p);
        if (rc) { std::printf("%-52s rmsnorm(k) rc=%d\n", tag, rc); ++g_fail; }
    }
    if (g.nq > 0) {
        std::vector<Buf*> o{qc, pos};
        std::vector<RadParam> p{Pi("M", g.M), Pi("head_dim", g.hd), Pi("n_head", g.nq),
                                Pi("n_head_kv", 0), Pf("theta", theta), Pf("scale", 1.0),
                                Ps("mode", "neox"), Pi("rotary_dim", g.rot)};
        int rc = run(rope_row, o, p);
        if (rc) { std::printf("%-52s rope(q) rc=%d\n", tag, rc); ++g_fail; }
    }
    {
        std::vector<Buf*> o{k_c, pos};
        std::vector<RadParam> p{Pi("M", g.M), Pi("head_dim", g.hd), Pi("n_head", g.nkv),
                                Pi("n_head_kv", 0), Pf("theta", theta), Pf("scale", 1.0),
                                Ps("mode", "neox"), Pi("rotary_dim", g.rot)};
        int rc = run(rope_row, o, p);
        if (rc) { std::printf("%-52s rope(k) rc=%d\n", tag, rc); ++g_fail; }
    }
    if (fuse_kv) {
        std::vector<Buf*> o{k_c, v, slot, cc};
        std::vector<RadParam> p{Pi("M", g.M), Pi("head_dim", g.hd), Pi("n_head_kv", g.nkv),
                                Pi("block_size", g.bs), Ps("kv_dtype", "bf16")};
        int rc = run(ks_row, o, p);
        if (rc) { std::printf("%-52s kv_store rc=%d\n", tag, rc); ++g_fail; }
    }

    /* ---- fused: same seeds, one launch */
    k_f->h = k_seed;
    std::memset(qf->h.data(), 0, qf->h.size());
    std::memset(cf->h.data(), 0xEE, cf->h.size());
    std::memset(gof->h.data(), 0, gof->h.size());
    std::vector<Buf*> fo;
    if (g.nq > 0) { fo.push_back(qg); } else { Buf none; none.present = false;
        /* k-only form: q_gate/q_w/q_out absent */ 
        keep.push_back(new Buf()); keep.back()->present = false; fo.push_back(keep.back()); }
    fo.push_back(k_f); fo.push_back(cs); fo.push_back(pos);
    if (g.nq > 0) fo.push_back(wq); else { keep.push_back(new Buf()); keep.back()->present = false; fo.push_back(keep.back()); }
    fo.push_back(wk);
    if (g.nq > 0) fo.push_back(qf); else { keep.push_back(new Buf()); keep.back()->present = false; fo.push_back(keep.back()); }
    fo.push_back(k_f);
    if (gated && g.nq > 0) fo.push_back(gof); else { keep.push_back(new Buf()); keep.back()->present = false; fo.push_back(keep.back()); }
    if (fuse_kv) { fo.push_back(v); fo.push_back(slot); fo.push_back(cf); }
    std::vector<RadParam> fp{Pi("M", g.M), Pi("head_dim", g.hd), Pi("n_head", g.nq),
                             Pi("n_head_kv", g.nkv), Pi("rot", g.rot), Ps("q_dtype", "bf16"),
                             Pf("eps", eps), Pf("wadd", wadd), Pi("cs_f32", g.cs_f32)};
    if (fuse_kv) { fp.push_back(Pi("block_size", g.bs)); fp.push_back(Ps("kv_dtype", "bf16")); }
    int rc = run(fused_row, fo, fp);
    if (getenv("QK_DIAG")) std::fprintf(stderr, "PHASE fused rc=%d %s\n", rc, tag);
    if (rc) { std::printf("%-52s FUSED launch rc=%d\n", tag, rc); ++g_fail;
        for (auto* b : keep) delete b; return; }

    /* ---- compare. Row 0 with pad: skip q/k bytes (defined-garbage divergence, see header),
     * still require untouched cache rows byte-equal. */
    if (g.nq > 0) {
        if (!pad) cmp("q_out", qc->h, qf->h);
        else {
            size_t rowb = (size_t)g.nq * g.hd * 2;
            if (!std::equal(qc->h.begin() + rowb, qc->h.end(), qf->h.begin() + rowb)) { ++g_fail; std::printf("  q_out(valid rows) differs\n"); } else ++g_pass;
        }
    }
    if (!pad) cmp("k_out", k_c->h, k_f->h);
    else {
        size_t rowb = (size_t)g.nkv * g.hd * 2;
        if (!std::equal(k_c->h.begin() + rowb, k_c->h.end(), k_f->h.begin() + rowb)) { ++g_fail; std::printf("  k_out(valid rows) differs\n"); } else ++g_pass;
    }
    if (fuse_kv) cmp("cache", cc->h, cf->h);
    if (getenv("QK_DIAG") && g_fail) {
        const uint16_t *qc16 = (const uint16_t*)qc->h.data(), *qf16 = (const uint16_t*)qf->h.data();
        std::printf("  DIAG pos:");
        for (int i = 0; i < g.M; ++i)
            std::printf(" %lld", g.pos_i64 ? (long long)((const int64_t*)pos->h.data())[i]
                                           : (long long)((const int32_t*)pos->h.data())[i]);
        std::printf("\n  DIAG q row elems (chain vs fused) tok head elem0:");
        for (int t = 0; t < g.M; ++t)
            std::printf(" [%d,%d]=(%04x,%04x)", t, 0,
                        qc16[(size_t)t * g.nq * g.hd], qf16[(size_t)t * g.nq * g.hd]);
        std::printf("\n  DIAG wk[0..1]=%04x %04x cs[co..]=%08x\n",
                    g.w_f32 ? ((const uint32_t*)wk->h.data())[0] : ((const uint16_t*)wk->h.data())[0],
                    g.w_f32 ? ((const uint32_t*)wk->h.data())[1] : ((const uint16_t*)wk->h.data())[1],
                    g.cs_f32 ? ((const uint32_t*)cs->h.data())[0] : 0u);
    }
    if (gated && g.nq > 0) {
        /* gate_out is the raw slice qg[tok][hd .. 2hd) per head: rebuild from the seed. */
        std::vector<uint8_t> want(gof->h.size());
        const uint16_t* src = (const uint16_t*)qg->h.data();
        uint16_t* dst = (uint16_t*)want.data();
        for (int t = 0; t < g.M; ++t)
            for (int h = 0; h < g.nq; ++h)
                for (int e = 0; e < g.hd; ++e)
                    dst[((size_t)t * g.nq + h) * g.hd + e] = src[((size_t)t * g.nq + h) * 2 * g.hd + g.hd + e];
        cmp("gate_out", want, gof->h);
    }
    std::printf("%-52s %s\n", tag, g_fail ? "(see MISMATCH above)" : "ok");
    if (getenv("QK_DIAG")) std::fprintf(stderr, "PHASE compare-done %s\n", tag);
    if (!getenv("QK_LEAK")) for (auto* b : keep) delete b;
    if (getenv("QK_DIAG")) std::fprintf(stderr, "PHASE freed %s\n", tag);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s <libr3d.so>\n", argv[0]); return 2; }
    Lib lib; lib.open(argv[1]);
    CK(hipSetDevice(0));

    /* QK_CASE picks one geometry for isolation; QK_REPEAT runs it N times to expose
     * nondeterminism (identical seeds every pass, so any flip is device-side). */
    const char* only = getenv("QK_CASE");
    const int reps = only ? atoi(getenv("QK_REPEAT") ? getenv("QK_REPEAT") : "1") : 1;
    for (int rep = 0; rep < reps; ++rep) {
    std::mt19937 rng(20261027u);
    auto pick = [&](int idx) { return !only || atoi(only) == idx; };
    int idx = 0;

    /* Geometry sweep: the compiled head widths, partial rotary, decode and prefill token
     * counts, both gain widths and both position widths. The cos/sin table stays f32 in every
     * identity case on purpose: the issuing architecture passes cs_f32=1, and a bf16 table
     * rounds the angles the chain computes in full f32 -- it is a valid kernel path, but it
     * is a different function, exactly as the kernel header says. */
    for (int hd : {64, 128, 256})
        for (int rot : {hd, hd / 2})
            for (int M : {1, 16})
                if (pick(idx)) case_run(lib, rng, Geo{hd, rot, M, 8, 2, 16, 1, 1, 0}, false, false, false);
                else { (void)rng; ++idx - idx; }
    if (!only) {
    for (int M : {1, 8})
        case_run(lib, rng, Geo{128, 128, M, 8, 2, 16, 1, 1, 0}, true, false, false);
    case_run(lib, rng, Geo{64, 32, 4, 8, 2, 16, 1, 0, 0}, true, false, false);   // bf16 gains
    case_run(lib, rng, Geo{64, 32, 4, 8, 2, 16, 1, 1, 1}, true, false, false);   // i64 pos
    case_run(lib, rng, Geo{128, 128, 8, 8, 2, 16, 1, 1, 0}, false, true, false); // gate_out
    case_run(lib, rng, Geo{128, 128, 8, 8, 2, 16, 1, 1, 0}, true, false, true);  // padded row 0
    case_run(lib, rng, Geo{128, 128, 8, 0, 4, 16, 1, 1, 0}, true, false, false); // k-only form
    case_run(lib, rng, Geo{256, 64, 2, 8, 2, 16, 1, 0, 1}, true, false, false);  // mixed widths
    } else {
        /* QK_CASE indices into the extras: 100=bf16g 101=i64 102=gate 103=pad 104=konly 105=mixed */
        int w = atoi(only);
        if (w == 100) case_run(lib, rng, Geo{64, 32, 4, 8, 2, 16, 1, 0, 0}, true, false, false);
        if (w == 101) case_run(lib, rng, Geo{64, 32, 4, 8, 2, 16, 1, 1, 1}, true, false, false);
        if (w == 105) case_run(lib, rng, Geo{256, 64, 2, 8, 2, 16, 1, 0, 1}, true, false, false);
        (void)idx;
    }
    }

    std::printf("\n%s: %d comparisons passed, %d failed\n",
                g_fail ? "FAIL" : "PASS", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
