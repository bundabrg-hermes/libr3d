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
#include "rad_plugin.h"      /* Q62: the fp8 store test's oracle IS the ABI encoder */

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
    /* Q62: op "kv_store" and "qk_norm_rope_gate_kv_store" each claim two rows (bf16,
     * fp8_e4m3); a differential case picks one by row name. */
    const RadKernelInfo* row_named(const char* name) {
        for (int i = 0; i < count(); ++i) {
            const RadKernelInfo* k = at(i);
            if (k && std::strcmp(k->name, name) == 0 && k->domain == RAD_DOMAIN_DEVICE) return k;
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

struct Geo { int hd, rot, M, nq, nkv, bs, cs_f32, w_f32, pos_i64, pitch; };

/* Copy the valid [M, ncols] window out of a pitched image into a dense buffer, for
 * comparing the fused side (pitched, like live) against the chain side (dense). */
static void compact(std::vector<uint8_t>& dense, const std::vector<uint8_t>& img,
                    long long M, long long pitch, long long ncols) {
    dense.assign((size_t)M * ncols * 2, 0);
    const uint16_t* s = (const uint16_t*)img.data();
    uint16_t* d = (uint16_t*)dense.data();
    for (long long t = 0; t < M; ++t)
        for (long long c = 0; c < ncols; ++c)
            d[t * ncols + c] = s[t * pitch + c];
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

/* One differential case: run the chain pipeline and the fused row on identical seeds and
 * compare every output byte for byte. `fuse_kv` selects the fold row; `gated` materializes
 * gate_out; `pad` makes row 0 a padded token (slot/position negative). */
static void case_run(Lib& lib, std::mt19937& rng, const Geo& g, bool fuse_kv,
                     bool gated, bool pad) {
    const int CS_ROWS = 64, nb = (g.M + g.bs - 1) / g.bs + 2;
    char tag[160];
    const int fails_before = g_fail;
    std::snprintf(tag, sizeof tag, "%s hd=%d rot=%d M=%d nq=%d nkv=%d cs=%s w=%s pos=%s%s%s",
                  fuse_kv ? "KV" : "plain", g.hd, g.rot, g.M, g.nq, g.nkv,
                  g.cs_f32 ? "f32" : "bf16", g.w_f32 ? "f32" : "bf16",
                  g.pos_i64 ? "i64" : "i32", gated ? " gate" : "", pad ? " pad" : "");

    std::vector<Buf*> keep;
    auto mkbuf = [&](uint32_t dt, std::vector<int64_t> sh) {
        Buf* b = new Buf(); b->init(dt, std::move(sh)); keep.push_back(b); return b;
    };
    const long long kcols = (long long)g.nkv * g.hd;
    const long long kpitch = kcols + (g.pitch ? 96 : 0);

    Buf* qg  = mkbuf(RAD_BF16, {g.M, (long long)g.nq * 2 * g.hd});
    Buf* k_c = mkbuf(RAD_BF16, {g.M, kcols});                              // chain: dense
    Buf* k_f = mkbuf(RAD_BF16, {g.M, kpitch});                             // fused: pitched image
    Buf* v_c = mkbuf(RAD_BF16, {g.M, kcols});                              // chain: dense
    Buf* v_f = mkbuf(RAD_BF16, {g.M, kpitch});                             // fused: pitched image
    Buf* wq  = mkbuf(g.w_f32 ? RAD_F32 : RAD_BF16, {g.hd});
    Buf* wk  = mkbuf(g.w_f32 ? RAD_F32 : RAD_BF16, {g.hd});
    Buf* pos = mkbuf(g.pos_i64 ? RAD_I64 : RAD_I32, {g.M});
    Buf* slot = mkbuf(RAD_I32, {g.M});
    Buf* cs  = mkbuf(g.cs_f32 ? RAD_F32 : RAD_BF16, {CS_ROWS, g.rot});
    Buf* qc  = mkbuf(RAD_BF16, {g.M, (long long)g.nq * g.hd});
    Buf* qf  = mkbuf(RAD_BF16, {g.M, (long long)g.nq * g.hd});
    Buf* cc  = mkbuf(RAD_BF16, {nb, g.nkv, g.bs, 2 * g.hd});
    const long long cfp = g.pitch ? 2 * g.hd + 96 : 2 * g.hd;         // fused cache image pitch
    Buf* cf  = mkbuf(RAD_BF16, {nb, g.nkv, g.bs, cfp});
    Buf* cs_f = mkbuf(g.cs_f32 ? RAD_F32 : RAD_BF16,
                      {CS_ROWS, g.pitch ? (long long)g.rot + 192 : (long long)g.rot});
    Buf* gof = mkbuf(RAD_BF16, {g.M, (long long)g.nq * g.hd});

    fill_bf16(rng, *qg, 1.0f); fill_bf16(rng, *k_c, 1.0f); fill_bf16(rng, *v_c, 1.0f);
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
    /* Seed a pitched image: poison everywhere, valid window from the dense seed. */
    auto seed_pitched = [&](Buf* img, const std::vector<uint8_t>& dense) {
        for (auto& b : img->h) b = 0xA5;
        const uint16_t* s = (const uint16_t*)dense.data();
        uint16_t* d = (uint16_t*)img->h.data();
        for (long long t = 0; t < g.M; ++t)
            for (long long c = 0; c < kcols; ++c) d[t * kpitch + c] = s[t * kcols + c];
    };
    seed_pitched(v_f, v_c->h);
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
        std::vector<Buf*> o{k_c, v_c, slot, cc};
        std::vector<RadParam> p{Pi("M", g.M), Pi("head_dim", g.hd), Pi("n_head_kv", g.nkv),
                                Pi("block_size", g.bs), Ps("kv_dtype", "bf16")};
        int rc = run(ks_row, o, p);
        if (rc) { std::printf("%-52s kv_store rc=%d\n", tag, rc); ++g_fail; }
    }

    /* ---- fused: same seeds, one launch, through pitched views exactly like live. The
     * cos/sin plane on this architecture is shared with the GDN prologue: a [rows, rot]
     * window of a wider allocation. The cache image is pitched too. Poison fills the
     * padding so a kernel that walked it would diverge. */
    seed_pitched(k_f, k_seed);
    {   /* copy the dense table into the pitched image */
        const size_t es = g.cs_f32 ? 4 : 2;
        for (auto& b : cs_f->h) b = 0xA5;
        const uint8_t* s = cs->h.data(); uint8_t* d = cs_f->h.data();
        for (long long r = 0; r < CS_ROWS; ++r)
            for (long long c = 0; c < g.rot; ++c)
                std::memcpy(d + (r * ((long long)g.rot + (g.pitch ? 192 : 0)) + c) * es,
                            s + (r * g.rot + c) * es, es);
    }
    Buf k_fv; k_fv.dtype = RAD_BF16;
    if (g.pitch) { k_fv.shape = {g.M, g.nkv, g.hd}; k_fv.strides = {kpitch, g.hd, 1}; }
    else         { k_fv.shape = {g.M, kcols};       k_fv.strides = {kpitch, 1}; }
    k_fv.alias = k_f;
    Buf v_fv; v_fv.dtype = RAD_BF16;
    if (g.pitch) { v_fv.shape = {g.M, g.nkv, g.hd}; v_fv.strides = {kpitch, g.hd, 1}; }
    else         { v_fv.shape = {g.M, kcols};       v_fv.strides = {kpitch, 1}; }
    v_fv.alias = v_f;
    Buf qg3; qg3.dtype = RAD_BF16; qg3.shape = {g.M, g.nq, 2 * g.hd};
    qg3.strides = {(long long)g.nq * 2 * g.hd, 2 * g.hd, 1}; qg3.alias = qg;  /* live rank-3 */
    Buf cs_fv; cs_fv.dtype = g.cs_f32 ? RAD_F32 : RAD_BF16; cs_fv.shape = {CS_ROWS, g.rot};
    cs_fv.strides = {(long long)g.rot + (g.pitch ? 192 : 0), 1}; cs_fv.alias = cs_f;
    Buf cf_v; cf_v.dtype = RAD_BF16; cf_v.shape = {nb, g.nkv, g.bs, 2 * g.hd};
    cf_v.strides = {g.nkv * g.bs * cfp, g.bs * cfp, cfp, 1}; cf_v.alias = cf;
    std::memset(qf->h.data(), 0, qf->h.size());
    std::memset(cf->h.data(), 0xEE, cf->h.size());
    std::memset(gof->h.data(), 0, gof->h.size());
    std::vector<Buf*> fo;
    if (g.nq > 0) { fo.push_back(g.pitch ? &qg3 : qg); } else {
        /* k-only form: q_gate/q_w/q_out absent */
        keep.push_back(new Buf()); keep.back()->present = false; fo.push_back(keep.back()); }
    fo.push_back(&k_fv); fo.push_back(&cs_fv); fo.push_back(pos);
    if (g.nq > 0) fo.push_back(wq); else { keep.push_back(new Buf()); keep.back()->present = false; fo.push_back(keep.back()); }
    fo.push_back(wk);
    if (g.nq > 0) fo.push_back(qf); else { keep.push_back(new Buf()); keep.back()->present = false; fo.push_back(keep.back()); }
    fo.push_back(&k_fv);
    if (gated && g.nq > 0) fo.push_back(gof); else { keep.push_back(new Buf()); keep.back()->present = false; fo.push_back(keep.back()); }
    if (fuse_kv) { fo.push_back(&v_fv); fo.push_back(slot); fo.push_back(&cf_v); }
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
    std::vector<uint8_t> kf_cmp;
    compact(kf_cmp, k_f->h, g.M, kpitch, kcols);
    if (!pad) cmp("k_out", k_c->h, kf_cmp);
    else {
        size_t rowb = (size_t)g.nkv * g.hd * 2;
        if (!std::equal(k_c->h.begin() + rowb, k_c->h.end(), kf_cmp.begin() + rowb)) { ++g_fail; std::printf("  k_out(valid rows) differs\n"); } else ++g_pass;
    }
    if (fuse_kv) {
        std::vector<uint8_t> cf_cmp(cc->h.size(), 0);
        const uint16_t* s = (const uint16_t*)cf->h.data();
        uint16_t* d = (uint16_t*)cf_cmp.data();
        for (long long b = 0; b < nb; ++b)
            for (long long h = 0; h < g.nkv; ++h)
                for (long long o = 0; o < g.bs; ++o)
                    for (long long e = 0; e < 2 * g.hd; ++e)
                        d[(((b * g.nkv + h) * g.bs + o) * 2 * g.hd) + e] =
                            s[b * g.nkv * g.bs * cfp + h * g.bs * cfp + o * cfp + e];
        cmp("cache", cc->h, cf_cmp);
    }
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
    std::printf("%-52s %s\n", tag, g_fail > fails_before ? "(see MISMATCH above)" : "ok");
    if (getenv("QK_DIAG")) std::fprintf(stderr, "PHASE compare-done %s\n", tag);
    if (!getenv("QK_LEAK")) for (auto* b : keep) delete b;
    if (getenv("QK_DIAG")) std::fprintf(stderr, "PHASE freed %s\n", tag);
}


/* ================= Q62: the fp8 (E4M3) store rows =================
 * The oracle is not the kernel's own arithmetic re-spelled: it is the ABI's host
 * rad_f32_to_fp8e4m3 (rad_plugin.h) -- the very encoder libref's stt() uses. Chain side:
 * qk_norm_rope_gate (prologue row) + kv_store_fp8 on its k_out, exactly what the engine
 * would run unfused. Fused side: qk_norm_rope_gate_kv_store_fp8 on pitched views, as
 * live hands them. Byte-identity is the law on both legs.
 * Row-count law (the dead-row bug class): M sweeps past one wave's worth of
 * (token,head) blocks (grid = M x (nq+nkv)), and slots scatter across MORE block rows
 * than a wave, so a row that silently never dispatched cannot hide. Slots include
 * negative and >= nb*bs: those cache bytes must stay poisoned at 0xA5. */
static void fp8_store_case(Lib& lib, std::mt19937& rng, int hd, int M, int nq, int nkv,
                           int bs, int pitch, double kscale, double vscale) {
    char tag[160];
    const int fails_before = g_fail;
    const int nb = 9;                                    /* fixed pool: slots scatter wide */
    std::snprintf(tag, sizeof tag, "F8-store hd=%d M=%d nq=%d nkv=%d bs=%d%s ks=%g vs=%g",
                  hd, M, nq, nkv, bs, pitch ? " pitch" : "", kscale, vscale);
    std::vector<Buf*> keep;
    auto mkbuf = [&](uint32_t dt, std::vector<int64_t> sh) {
        Buf* b = new Buf(); b->init(dt, std::move(sh)); keep.push_back(b); return b;
    };
    const long long kcols = (long long)nkv * hd;
    const long long kpitch = kcols + (pitch == 1 ? 96 : pitch == 2 ? 4 : 0);
    const long long cfp = 2LL * hd + (pitch == 1 ? 96 : pitch == 2 ? 4 : 0);

    Buf* k   = mkbuf(RAD_BF16, {M, kpitch});
    Buf* v   = mkbuf(RAD_BF16, {M, kpitch});
    Buf* slot = mkbuf(RAD_I32, {M});
    Buf* cache = mkbuf(RAD_F8E4M3, {nb, nkv, bs, cfp});
    for (size_t i = 0; i < k->n(); ++i) {                    /* amp spread: normals, big (clamp), tiny (flush) */
        float amp = (i % 7 == 0) ? 600.0f : (i % 11 == 0) ? 0.001f : 1.0f;
        std::uniform_real_distribution<float> u(-amp, amp);
        float f = u(rng); uint32_t u32; std::memcpy(&u32, &f, 4);
        ((uint16_t*)k->h.data())[i] = (uint16_t)(u32 >> 16);
        float f2 = u(rng) * (amp * 0.7f); std::memcpy(&u32, &f2, 4);
        ((uint16_t*)v->h.data())[i] = (uint16_t)(u32 >> 16);
    }
    std::memset(cache->h.data(), 0xA5, cache->h.size());
    /* Slots: a BIJECTION (coprime stride mod the pool) over valid slots, so every slot
     * has exactly one writer and the oracle is order-free; plus dedicated padding (-1)
     * and out-of-range (>= nb*bs) rows that must leave their bytes at the poison. */
    const int32_t total = (int32_t)(nb * bs);
    for (int i = 0; i < M; ++i) {
        int32_t s;
        if (i % 13 == 5) s = -1;
        else if (i % 17 == 9) s = (int32_t)(total + (i % bs));
        else s = (int32_t)((i * 5) % total);
        ((int32_t*)slot->h.data())[i] = s;
    }

    const RadKernelInfo* ks8 = lib.row_named("kv_store_fp8");
    if (!ks8) { std::printf("%-52s no kv_store_fp8 row\n", tag); ++g_fail; for (auto* b : keep) delete b; return; }

    /* ---- direct: kv_store_fp8 against the ABI-encoder oracle, byte for byte ---- */
    {
        std::vector<Buf*> o{ k, v, slot, cache };
        std::vector<RadParam> p{ Pi("head_dim", hd), Pi("block_size", bs),
                                 Pi("n_head_kv", nkv), Ps("kv_dtype", "fp8_e4m3"),
                                 Pf("k_scale", kscale), Pf("v_scale", vscale) };
        int rc = run(ks8, o, p);
        if (rc) { std::printf("%-52s kv_store_fp8 rc=%d\n", tag, rc); ++g_fail; }
        const double kmul = 1.0 / kscale, vmul = 1.0 / vscale;
        const uint16_t* kp = (const uint16_t*)k->h.data();
        const uint16_t* vp = (const uint16_t*)v->h.data();
        const int32_t*  sl = (const int32_t*)slot->h.data();
        uint8_t* cp = (uint8_t*)cache->h.data();
        std::vector<uint8_t> want(cache->h.size(), 0xA5);
        for (int t = 0; t < M; ++t) {
            const int32_t s = sl[t];
            if (s < 0 || s / bs >= nb) continue;
            const int64_t b = s / bs, off = s % bs;
            for (int h = 0; h < nkv; ++h) {
                uint8_t* dst = cp + (((b * nkv + h) * bs + off) * cfp);
                uint8_t* wdst = want.data() + (((b * nkv + h) * bs + off) * 2 * hd);
                for (int d = 0; d < hd; ++d) {
                    const uint16_t kb = kp[(size_t)t * kpitch + (size_t)h * hd + d];
                    const uint16_t vb = vp[(size_t)t * kpitch + (size_t)h * hd + d];
                    float kx, vx;   /* BIT reinterpret (a C float cast converts numerically!) */
                    { const uint32_t ku = (uint32_t)kb << 16, vu = (uint32_t)vb << 16;
                      std::memcpy(&kx, &ku, 4); std::memcpy(&vx, &vu, 4); }
                    const uint8_t kcode = rad_f32_to_fp8e4m3(kx * (float)kmul);
                    const uint8_t vcode = rad_f32_to_fp8e4m3(vx * (float)vmul);
                    wdst[d] = kcode; wdst[hd + d] = vcode;
                }
            }
        }
        /* Compare byte-for-byte over the dense footprint (pitch columns must stay 0xA5). */
        for (int64_t b = 0; b < nb; ++b)
            for (int64_t h = 0; h < nkv; ++h)
                for (int64_t off = 0; off < bs; ++off)
                    for (int64_t e = 0; e < 2 * hd; ++e) {
                        const uint8_t got = cp[((b * nkv + h) * bs + off) * cfp + e];
                        const uint8_t exp = want[((b * nkv + h) * bs + off) * 2 * hd + e];
                        if (got != exp) {
                            ++g_fail;
                            if (g_fail - fails_before < 6)
                                std::printf("  MISMATCH b=%lld h=%lld off=%lld e=%lld got=%02x exp=%02x\n",
                                            b, h, off, e, got, exp);
                        }
                    }
        for (int64_t b = 0; b < nb; ++b)
            for (int64_t h = 0; h < nkv; ++h)
                for (int64_t off = 0; off < bs; ++off)
                    for (int64_t e = 2 * hd; e < cfp; ++e)
                        if (cp[((b * nkv + h) * bs + off) * cfp + e] != 0xA5) {
                            ++g_fail;
                            if (g_fail - fails_before < 2) std::printf("  REDZONE walked at e=%lld\n", e);
                        }
        if (!g_fail) ++g_pass;
    }
    if (getenv("QK_DIAG")) std::fprintf(stderr, "PHASE f8-direct %s\n", tag);
    for (auto* b : keep) delete b;
    std::printf("%-52s %s\n", tag, g_fail > fails_before ? "MISMATCH (above)" : "ok");
}

/* Fused-vs-chain fp8: qk_norm_rope_gate + kv_store_fp8 vs qk_norm_rope_gate_kv_store_fp8,
 * identical seeds, pitched views, byte-identical cache. hd=256 mirrors the 27B. */
static void fp8_fused_case(Lib& lib, std::mt19937& rng, int hd, int rot, int M, int nq,
                           int nkv, int bs, double kscale, double vscale) {
    char tag[160];
    const int fails_before = g_fail;
    const int nb = 9;
    std::snprintf(tag, sizeof tag, "F8-fused hd=%d rot=%d M=%d nq=%d nkv=%d bs=%d ks=%g",
                  hd, rot, M, nq, nkv, bs, kscale);
    const int CS_ROWS = 96;
    std::vector<Buf*> keep;
    auto mkbuf = [&](uint32_t dt, std::vector<int64_t> sh) {
        Buf* b = new Buf(); b->init(dt, std::move(sh)); keep.push_back(b); return b;
    };
    const long long kcols = (long long)nkv * hd;
    const long long kpitch = kcols + 96;                 /* pitched everywhere: live shape */

    Buf* qg   = mkbuf(RAD_BF16, {M, (long long)nq * 2 * hd});
    Buf* kc   = mkbuf(RAD_BF16, {M, kcols});             /* chain k, dense */
    Buf* kf   = mkbuf(RAD_BF16, {M, kpitch});            /* fused k, pitched image */
    Buf* vc   = mkbuf(RAD_BF16, {M, kcols});
    Buf* vf   = mkbuf(RAD_BF16, {M, kpitch});
    Buf* wq   = mkbuf(RAD_BF16, {hd});
    Buf* wk   = mkbuf(RAD_BF16, {hd});
    Buf* pos  = mkbuf(RAD_I32, {M});
    Buf* slot = mkbuf(RAD_I32, {M});
    Buf* cs   = mkbuf(RAD_F32, {CS_ROWS, rot});
    Buf* qco  = mkbuf(RAD_BF16, {M, (long long)nq * hd});
    Buf* qfo  = mkbuf(RAD_BF16, {M, (long long)nq * hd});
    Buf* cchain = mkbuf(RAD_F8E4M3, {nb, nkv, bs, 2LL * hd});
    Buf* cfused = mkbuf(RAD_F8E4M3, {nb, nkv, bs, kpitch == kcols ? 2LL * hd : 2LL * hd + 96});
    const long long cfp = cfused->shape[3];

    fill_bf16(rng, *qg, 1.0f); fill_bf16(rng, *kc, 1.0f); fill_bf16(rng, *vc, 1.0f);
    fill_bf16(rng, *wq, 0.5f); fill_bf16(rng, *wk, 0.5f);
    for (size_t i = 0; i < cs->n(); ++i) ((float*)cs->h.data())[i] =
        std::uniform_real_distribution<float>(-1.f, 1.f)(rng);
    for (int i = 0; i < M; ++i) {
        ((int32_t*)pos->h.data())[i] = (int)(rng() % (unsigned)(CS_ROWS - 1)) + 1;
        int32_t s;
        if (i % 13 == 5) s = -1;
        else if (i % 17 == 9) s = (int32_t)(nb * bs + (i % bs));
        else s = (int32_t)((i * 5) % (nb * bs));       /* bijection: order-free oracle */
        ((int32_t*)slot->h.data())[i] = s;
    }
    std::memset(cchain->h.data(), 0xA5, cchain->h.size());
    std::memset(cfused->h.data(), 0xA5, cfused->h.size());

    const RadKernelInfo* pro  = lib.row_named("qk_norm_rope_gate");
    const RadKernelInfo* ks8  = lib.row_named("kv_store_fp8");
    const RadKernelInfo* fus8 = lib.row_named("qk_norm_rope_gate_kv_store_fp8");
    if (!pro || !ks8 || !fus8) { std::printf("%-52s missing row\n", tag); ++g_fail;
        for (auto* b : keep) delete b; return; }

    auto seed_pitched = [&](Buf* img, const std::vector<uint8_t>& dense) {
        for (auto& b : img->h) b = 0xA5;
        const uint16_t* s = (const uint16_t*)dense.data();
        uint16_t* d = (uint16_t*)img->h.data();
        for (long long t = 0; t < M; ++t)
            for (long long c = 0; c < kcols; ++c) d[t * kpitch + c] = s[t * kcols + c];
    };

    /* The fused side must see the ORIGINAL k/v, not the chain prologue's in-place
     * rewrite of kc: snapshot before the chain runs. */
    const std::vector<uint8_t> kc_seed = kc->h;

    /* ---- chain ---- */
    {
        std::vector<Buf*> o{ qg, kc, cs, pos, wq, wk, qco, kc };
        std::vector<RadParam> p{ Pi("M", M), Pi("head_dim", hd), Pi("n_head", nq),
                                 Pi("n_head_kv", nkv), Pi("rot", rot), Ps("q_dtype", "bf16"),
                                 Pf("eps", 1e-6), Pf("wadd", 0.0), Pi("cs_f32", 1) };
        int rc = run(pro, o, p);
        if (rc) { std::printf("%-52s chain prologue rc=%d\n", tag, rc); ++g_fail; }
        seed_pitched(vf, vc->h);
        Buf vf_c2; vf_c2.dtype = RAD_BF16; vf_c2.shape = {M, nkv, hd};   /* pitched like live */
        vf_c2.strides = {kpitch, hd, 1}; vf_c2.alias = vf;
        std::vector<Buf*> o2{ kc, &vf_c2, slot, cchain };    /* k = dense k_out, v pitched */
        std::vector<RadParam> p2{ Pi("head_dim", hd), Pi("block_size", bs),
                                  Pi("n_head_kv", nkv), Ps("kv_dtype", "fp8_e4m3"),
                                  Pf("k_scale", kscale), Pf("v_scale", vscale) };
        rc = run(ks8, o2, p2);
        if (rc) { std::printf("%-52s chain kv_store_fp8 rc=%d\n", tag, rc); ++g_fail; }
    }
    if (getenv("QK_DIAG")) std::fprintf(stderr, "PHASE f8-chain %s\n", tag);

    /* ---- fused: pitched k/v/cache views, cache wider than 2*hd ---- */
    seed_pitched(kf, kc_seed); seed_pitched(vf, vc->h);
    Buf kf_v; kf_v.dtype = RAD_BF16; kf_v.shape = {M, nkv, hd};
    kf_v.strides = {kpitch, hd, 1}; kf_v.alias = kf;
    Buf vf_v; vf_v.dtype = RAD_BF16; vf_v.shape = {M, nkv, hd};
    vf_v.strides = {kpitch, hd, 1}; vf_v.alias = vf;
    Buf cf_v; cf_v.dtype = RAD_F8E4M3; cf_v.shape = {nb, nkv, bs, 2 * hd};
    cf_v.strides = {nkv * bs * cfp, bs * cfp, cfp, 1}; cf_v.alias = cfused;
    std::memset(qfo->h.data(), 0, qfo->h.size());
    {
        keep.push_back(new Buf()); keep.back()->present = false;   /* absent gate_out slot */
        std::vector<Buf*> o{ qg, &kf_v, cs, pos, wq, wk, qfo, &kf_v, keep.back(),
                             &vf_v, slot, &cf_v };
        std::vector<RadParam> p{ Pi("M", M), Pi("head_dim", hd), Pi("n_head", nq),
                                 Pi("n_head_kv", nkv), Pi("rot", rot), Ps("q_dtype", "bf16"),
                                 Pf("eps", 1e-6), Pf("wadd", 0.0), Pi("cs_f32", 1),
                                 Pi("block_size", bs), Ps("kv_dtype", "fp8_e4m3"),
                                 Pf("k_scale", kscale), Pf("v_scale", vscale) };
        int rc = run(fus8, o, p);
        if (rc) { std::printf("%-52s fused rc=%d\n", tag, rc); ++g_fail;
            for (auto* b : keep) delete b; return; }
    }
    if (getenv("QK_DIAG")) std::fprintf(stderr, "PHASE f8-fused %s\n", tag);

    /* ---- compare caches byte for byte (chain dense vs fused pitched) ---- */
    {
        const uint8_t* s = cfused->h.data();
        for (int64_t b = 0; b < nb; ++b)
            for (int64_t h = 0; h < nkv; ++h)
                for (int64_t off = 0; off < bs; ++off)
                    for (int64_t e = 0; e < 2 * hd; ++e) {
                        const uint8_t got = s[((b * nkv + h) * bs + off) * cfp + e];
                        const uint8_t exp = cchain->h[((b * nkv + h) * bs + off) * 2 * hd + e];
                        if (got != exp) {
                            ++g_fail;
                            if (g_fail - fails_before < 6) {
                                const uint8_t* kp = cfused->h.data();
                                std::printf("  MISMATCH b=%lld h=%lld off=%lld e=%lld got=%02x exp=%02x"
                                            " | fused[0..7]=%02x %02x %02x %02x chain[0..3]=%02x %02x %02x %02x\n",
                                            b, h, off, e, got, exp,
                                            kp[0],kp[1],kp[2],kp[3],
                                            cchain->h[0],cchain->h[1],cchain->h[2],cchain->h[3]);
                            }
                        }
                    }
        if (!g_fail) ++g_pass;
    }
    std::printf("%-52s %s\n", tag, g_fail > fails_before ? "MISMATCH (above)" : "ok");
    for (auto* b : keep) delete b;
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
                if (pick(idx)) case_run(lib, rng, Geo{hd, rot, M, 8, 2, 16, 1, 1, 0, 0}, false, false, false);
                else (void)rng;
    /* Q62 fp8 store coverage: the compiled widths, past-one-wave row counts, the
     * alignment-split (pitch) pair, scales off the identity, and the fused-vs-chain
     * differential at the 27B's shape (hd=256, nq=8, nkv=2). */
    if (!only || atoi(only) >= 200) {
        struct F8Case { int hd, M, nq, nkv, bs, pitch; double ks, vs; };
        static const F8Case f8[] = {
            { 256, 131, 8, 2, 16, 0, 1.0, 1.0 },
            { 256,  64, 8, 2, 16, 1, 1.0, 1.0 },
            { 256,  17, 8, 2, 16, 2, 2.0, 0.5 },      /* odd pitch: the scalar walk */
            { 128, 256, 8, 2, 16, 0, 1.0, 1.0 },
            { 128,  33, 8, 2, 16, 1, 0.25, 4.0 },
            {  64,  64, 8, 2, 16, 2, 1.0, 1.0 },
        };
        static const F8Case f8f[] = {
            { 256, 131, 8, 2, 16, 0, 1.0, 1.0 },
            { 256,  64, 8, 2, 16, 0, 2.0, 2.0 },
            { 128, 131, 8, 2, 16, 0, 1.0, 1.0 },
            {  64,  64, 8, 2, 16, 0, 0.5, 0.5 },
        };
        if (!only || atoi(only) == 200)
            for (auto& c : f8)  fp8_store_case (lib, rng, c.hd, c.M, c.nq, c.nkv, c.bs, c.pitch, c.ks, c.vs);
        if (!only || atoi(only) == 201)
            for (auto& c : f8f) fp8_fused_case (lib, rng, c.hd, c.hd, c.M, c.nq, c.nkv, c.bs, c.ks, c.vs);
    }
    if (!only) {
    for (int M : {1, 8})
        case_run(lib, rng, Geo{128, 128, M, 8, 2, 16, 1, 1, 0, M == 8}, true, false, false);
    case_run(lib, rng, Geo{64, 32, 4, 8, 2, 16, 1, 0, 0, 0}, true, false, false);   // bf16 gains
    case_run(lib, rng, Geo{64, 32, 4, 8, 2, 16, 1, 1, 1, 1}, true, false, false);   // i64 pos, pitched
    case_run(lib, rng, Geo{128, 128, 8, 8, 2, 16, 1, 1, 0, 0}, false, true, false); // gate_out
    case_run(lib, rng, Geo{128, 128, 8, 8, 2, 16, 1, 1, 0, 1}, true, false, true);  // padded row 0
    case_run(lib, rng, Geo{128, 128, 8, 0, 4, 16, 1, 1, 0, 0}, true, false, false); // k-only form
    case_run(lib, rng, Geo{256, 64, 2, 8, 2, 16, 1, 0, 1, 1}, true, false, false);  // mixed widths, pitched
    } else {
        /* QK_CASE indices into the extras: 100=bf16g 101=i64 102=gate 103=pad 104=konly 105=mixed */
        int w = atoi(only);
        if (w == 100) case_run(lib, rng, Geo{64, 32, 4, 8, 2, 16, 1, 0, 0, 0}, true, false, false);
        if (w == 101) case_run(lib, rng, Geo{64, 32, 4, 8, 2, 16, 1, 1, 1, 1}, true, false, false);
        if (w == 105) case_run(lib, rng, Geo{256, 64, 2, 8, 2, 16, 1, 0, 1, 1}, true, false, false);
        (void)idx;
    }
    }

    std::printf("\n%s: %d comparisons passed, %d failed\n",
                g_fail ? "FAIL" : "PASS", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
