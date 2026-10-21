/* replay_dump.cpp -- replay a captured serve launch against libref's host row.
 *
 * Pair with src/r3d_dump.hip: with R3D_DUMP=<dir> set in the serve environment, the
 * first R3D_DUMP_MAX launches per suspect row land in <dir>/NNNN_<row>/ (manifest.tsv,
 * pre.bin, post.bin). This tool rebuilds that exact launch on host buffers -- operand
 * views and all (elements are scattered back to the dumped strides) -- runs libref's
 * host row for the same op, and diffs the reference bytes against what the card
 * actually wrote under the engine.
 *
 *   ALL EQUAL   the row agrees with ref ON THE ENGINE'S OWN INPUTS: the bug is not here.
 *   MISMATCH    the row lies on real inputs: element values name the kind of lie.
 *   SANITY      an operand was truncated (weights too big to capture): only the argmax
 *               report runs -- enough to see whether a big output is garbage or shaped.
 *
 * Build (in radiance-dev, host-only, no card needed for ref -- but hipcc links hip):
 *   hipcc -O2 -I/opt/radiance/include/radiance tools/replay_dump.cpp -ldl -o /tmp/replay_dump
 * Run:
 *   /tmp/replay_dump <dumpdir>... <libr3d.so> <libref.so>
 */
#include <dlfcn.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "rad_abi.h"
#include <hip/hip_runtime.h>

typedef int (*CountFn)();
typedef const RadKernelInfo* (*AtFn)(int);

struct Buf { std::vector<uint8_t> h; };

static size_t esz(uint32_t dt) {
    switch (dt) {
        case RAD_F32: case RAD_I32: case RAD_U32: return 4;
        case RAD_I64:                              return 8;
        case RAD_F16: case RAD_BF16: case RAD_I16: return 2;
        case RAD_F8E4M3: case RAD_F8E5M2: case RAD_I8: case RAD_U8: case RAD_BOOL: return 1;
        default: return 4;
    }
}

struct Dump {
    std::string row, dir;
    int rc = 0, n_t = 0, n_p = 0;
    struct T { uint32_t dt, rank; std::vector<int64_t> sh, st; size_t bytes; int trunc, nullp; };
    std::vector<T> ts;
    struct P { int kind; std::string key; long long iv, ih; std::string sv; double dv; };
    std::vector<P> ps;
    std::vector<uint8_t> pre, post;
};

static bool load(const std::string& dir, Dump& d) {
    d.dir = dir;
    FILE* f = std::fopen((dir + "/manifest.tsv").c_str(), "r");
    if (!f) { std::printf("%s: no manifest\n", dir.c_str()); return false; }
    char line[4096];
    while (std::fgets(line, sizeof line, f)) {
        std::vector<std::string> t; char* save = nullptr;
        for (char* s = strtok_r(line, "\t\n", &save); s; s = strtok_r(nullptr, "\t\n", &save)) t.push_back(s);
        if (t.empty()) continue;
        if (t[0] == "H" && t.size() >= 5) { d.row = t[1]; d.rc = std::atoi(t[2].c_str());
                                            d.n_t = std::atoi(t[3].c_str()); d.n_p = std::atoi(t[4].c_str()); }
        else if (t[0] == "P" && t.size() >= 7) { Dump::P p; p.kind = std::atoi(t[1].c_str()); p.key = t[2];
            p.iv = std::atoll(t[3].c_str()); p.ih = std::atoll(t[4].c_str()); p.sv = t[5]; p.dv = std::atof(t[6].c_str());
            d.ps.push_back(p); }
        else if (t[0] == "T" && t.size() >= 6) {
            Dump::T x; x.dt = (uint32_t)std::stoul(t[2]); x.rank = (uint32_t)std::stoul(t[3]);
            size_t k = 4; for (; k < 4 + x.rank; ++k) x.sh.push_back(std::atoll(t[k].c_str()));
            for (; k < 4 + 2 * (size_t)x.rank; ++k) x.st.push_back(std::atoll(t[k].c_str()));
            x.bytes = (size_t)std::strtoull(t[k++].c_str(), nullptr, 10);
            x.trunc = k < t.size() ? std::atoi(t[k++].c_str()) : 0;
            x.nullp = k < t.size() ? std::atoi(t[k++].c_str()) : 0;
            d.ts.push_back(x); }
    }
    std::fclose(f);
    if (d.row.empty() || (int)d.ts.size() != d.n_t) { std::printf("%s: bad manifest\n", dir.c_str()); return false; }
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<uint8_t>& v = pass ? d.post : d.pre;
        FILE* g = std::fopen((dir + (pass ? "/post.bin" : "/pre.bin")).c_str(), "rb");
        if (!g) { if (pass) { std::printf("%s: no post.bin (row refused: rc=%d)\n", dir.c_str(), d.rc); continue; }
                  std::printf("%s: no pre.bin\n", dir.c_str()); return false; }
        v.resize(0);
        uint8_t tmp[65536]; size_t n;
        while ((n = std::fread(tmp, 1, sizeof tmp, g)) > 0) v.insert(v.end(), tmp, tmp + n);
        std::fclose(g);
    }
    return true;
}

/* The dump is the RAW span of memory at the tensor's data pointer (the exact region a
 * strided kernel touches). Host replay rebuilds that region verbatim; ref indexes it
 * with the same strides, so a whole-region memcmp is the comparison. */
static bool contig_at(const Dump::T& x) {
    int64_t s = 1;
    for (int64_t r = (int64_t)x.rank - 1; r >= 0; --r) {
        if (x.rank && x.st[r] != s) return false;
        s *= x.sh[r];
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) { std::printf("usage: %s <dumpdir>... <libr3d.so> <libref.so>\n", argv[0]); return 2; }
    void* hr = dlopen(argv[argc - 2], RTLD_NOW | RTLD_LOCAL);
    void* hf = dlopen(argv[argc - 1], RTLD_NOW | RTLD_LOCAL);
    if (!hr || !hf) { std::printf("dlopen: %s\n", dlerror()); return 2; }
    /* R3D_REPLAY_DEV=1 replays the DEVICE row (libr3d, the row serve selected) on a real
     * card instead of libref's host row: the same operands, the same views, the same
     * params, minus the engine. "serve wrote nothing" vs "the row is wrong" split here. */
    int dev_mode = std::getenv("R3D_REPLAY_DEV") != nullptr;
    void* htab = dev_mode ? hr : hf;
    AtFn atf = (AtFn)dlsym(htab, "rad_kernel_at");
    CountFn cntf = (CountFn)dlsym(htab, "rad_kernel_count");
    if (!atf || !cntf) { std::printf("kernel table missing\n"); return 2; }

    int fails = 0, total = 0;
    for (int i = 1; i < argc - 2; ++i) {
        Dump d; if (!load(argv[i], d)) { ++fails; continue; }
        if (std::getenv("R3D_REPLAY_DEBUG")) {
            std::printf("  PARSE row=%s n_t=%d ts=%zu n_p=%zu ps=%zu\n", d.row.c_str(), d.n_t,
                        d.ts.size(), d.n_p, d.ps.size());
            size_t o = 0;
            for (size_t t = 0; t < d.ts.size(); ++t) {
                std::printf("   ts[%zu] dt=%u rank=%u sh0=%lld bytes=%zu (off=%zu)\n", t, d.ts[t].dt,
                            d.ts[t].rank, (long long)d.ts[t].sh[0], d.ts[t].bytes, o);
                o += d.ts[t].bytes;
            }
        }
        /* op := row name minus the dtype suffix the libr3d table carries */
        std::string op = d.row;
        for (const char* suf : {"_bf16", "_e4m3", "_f32", "_f16"}) {
            size_t L = std::strlen(suf);
            if (op.size() > L && op.compare(op.size() - L, L, suf) == 0) { op.erase(op.size() - L); break; }
        }
        const RadKernelInfo* rk = nullptr;
        const int want_domain = dev_mode ? RAD_DOMAIN_DEVICE : RAD_DOMAIN_HOST;
        for (int k = 0; k < cntf(); ++k) {
            const RadKernelInfo* kk = atf(k);
            if (kk && !std::strcmp(kk->op, op.c_str()) && kk->domain == want_domain) { rk = kk; break; }
        }
        bool truncated = false;
        for (auto& x : d.ts) if (x.trunc) truncated = true;
        std::printf("\n=== %s (row %s, rc=%d)%s\n", argv[i], d.row.c_str(), d.rc,
                    rk ? "" : "  [libref has no host row]");
        fails = 0; int unver = 0;   /* per-directory verdict scope: a global fails
                                     counter silently suppressed every ALL EQUAL
                                     after the first mismatch (observed 2026-10-21) */
        if (!rk || truncated || d.post.empty()) {
            /* sanity: argmax of every large f32/bf16 output captured */
            size_t off = 0;
            for (size_t t = 0; t < d.ts.size(); ++t) {
                const Dump::T& x = d.ts[t];
                const size_t this_off = off;
                off += x.bytes;
                size_t els = x.bytes / esz(x.dt);
                if (x.nullp || els < 50000) continue;
                double best = -1e300; size_t bi = 0;
                for (size_t e = 0; e < els; ++e) {
                    const uint8_t* p = d.post.data() + this_off + e * esz(x.dt);
                    double v = 0;
                    if (x.dt == RAD_F32) { float f; std::memcpy(&f, p, 4); v = f; }
                    else if (x.dt == RAD_BF16) { uint32_t u = (uint32_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8)) << 16; float f; std::memcpy(&f, &u, 4); v = f; }
                    else continue;
                    if (!(v <= best)) { best = v; bi = e; if (!(v == v)) break; }
                }
                std::printf("  operand %zu: n=%zu argmax=%zu val=%.4f%s\n", t, els, bi, best,
                            (best != best) ? "  <-- NaN" : "");
            }
            std::printf("  SANITY only (%s)\n", truncated ? "operand truncated" : rk ? "no post state" : "no ref row");
            continue;
        }
        std::vector<std::vector<uint8_t>> dense(d.ts.size());
        std::vector<RadTensor> th(d.ts.size());
        std::vector<RadParam> tp(d.ps.size());
        std::vector<std::string> keep(d.ps.size());
        for (size_t t = 0; t < d.ts.size(); ++t) {
            const Dump::T& x = d.ts[t];
            size_t off = 0; for (size_t q = 0; q < t; ++q) off += d.ts[q].bytes;
            if (!x.nullp && x.bytes) dense[t].assign(d.pre.begin() + off, d.pre.begin() + off + x.bytes);
            th[t] = RadTensor{};
            th[t].dtype = x.dt; th[t].rank = x.rank;
            for (uint32_t r = 0; r < x.rank; ++r) { th[t].shape[r] = x.sh[r]; th[t].stride[r] = x.st[r]; }
            th[t].data = (!x.nullp && x.bytes) ? (void*)dense[t].data() : nullptr;
            if (!th[t].data) { th[t].rank = 1; th[t].shape[0] = 1; th[t].stride[0] = 1; th[t].dtype = RAD_F32; }
        }
        for (size_t q = 0; q < d.ps.size(); ++q) {
            keep[q] = d.ps[q].sv == "-" ? "" : d.ps[q].sv;
            tp[q] = RadParam{ d.ps[q].key.c_str(), d.ps[q].kind, d.ps[q].iv, d.ps[q].ih,
                              d.ps[q].kind == RAD_P_STR ? keep[q].c_str() : nullptr, d.ps[q].dv };
        }
        RadArgs h{}; h.t = th.data(); h.n_t = (int)th.size(); h.p = tp.data(); h.n_p = (int)tp.size();
        int rc;
        {   /* dev mode: the operand spans ride real device buffers, exactly as in the engine */
            std::vector<void*> dv(th.size(), nullptr);
            if (dev_mode) {
                for (size_t t = 0; t < th.size(); ++t) {
                    if (!th[t].data) continue;
                    if (hipMalloc(&dv[t], d.ts[t].bytes) != hipSuccess || !dv[t]) {
                        std::printf("  hipMalloc(%zu) failed\n", d.ts[t].bytes); return 3;
                    }
                    if (hipMemcpy(dv[t], dense[t].data(), d.ts[t].bytes,
                                  hipMemcpyHostToDevice) != hipSuccess) return 3;
                    th[t].data = dv[t];
                }
            }
            rc = rk->launch(&h, (RadStream)nullptr);
            std::printf("  %s launch rc=%d\n", dev_mode ? "dev" : "ref", rc);
            if (!dev_mode) {
                /* libref host rows may hand work to internal threads and return early:
                 * wait until every operand buffer is byte-stable before witnessing it.
                 * Without this the post-state read is a torn race (bogus diffs, segfaults). */
                std::vector<uint64_t> prev(dense.size(), ~0ull);
                int idle = 0;
                for (int round = 0; round < 150 && idle < 5; ++round) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    bool same = true;
                    for (size_t t = 0; t < dense.size(); ++t) {
                        uint64_t hsh = 1469598103934665603ull;
                        for (size_t b = 0; b < dense[t].size(); b += 64) {
                            hsh ^= dense[t][b]; hsh *= 1099511628211ull;
                        }
                        if (hsh != prev[t]) same = false;
                        prev[t] = hsh;
                    }
                    idle = same ? idle + 1 : 0;
                }
                if (idle < 5) std::printf("  WARNING: buffers still mutating after 30s\n");
            }
            if (dev_mode) {
                hipDeviceSynchronize();
                for (size_t t = 0; t < th.size(); ++t) {
                    if (!dv[t]) continue;
                    if (hipMemcpy(dense[t].data(), dv[t], d.ts[t].bytes,
                                  hipMemcpyDeviceToHost) != hipSuccess) return 3;
                    hipFree(dv[t]);
                }
            }
        }
        {   /* the post-launch buffers: the third witness, on disk */
            std::string p = std::string(argv[i]) + (dev_mode ? "/dev.bin" : "/ref.bin");
            FILE* f = std::fopen(p.c_str(), "wb");
            if (f) { for (size_t t = 0; t < dense.size(); ++t)
                         if (!dense[t].empty()) std::fwrite(dense[t].data(), 1, dense[t].size(), f);
                     std::fclose(f); }
        }
        if (rc != RAD_OK) { ++fails; continue; }
        size_t off = 0;
        for (size_t t = 0; t < d.ts.size(); ++t) {
            const Dump::T& x = d.ts[t];
            const size_t this_off = off;
            off += x.bytes;
            if (x.nullp || !x.bytes) continue;
            if (dense[t].size() < x.bytes) { std::printf("  operand %zu: dense shorter than dump\n", t); ++fails; continue; }
            if (std::memcmp(dense[t].data(), d.post.data() + this_off, x.bytes) == 0) continue;
            /* Vacuous-diff guard: a libref host row that silently declines the live
             * geometry leaves its outputs byte-equal to pre. Diffing the card's real
             * write against ref's non-write manufactures a MISMATCH. Label it. */
            if (std::memcmp(dense[t].data(), d.pre.data() + this_off, x.bytes) == 0 &&
                std::memcmp(d.post.data() + this_off, d.pre.data() + this_off, x.bytes) != 0) {
                std::printf("  operand %zu: REF NO-WRITE (card wrote %zu B) -> UNVERIFIABLE row/geometry\n",
                            t, x.bytes);
                ++unver; continue;
            }
            /* first-witness hex: dense vs in-memory post vs fresh reads from both files */
            {
                unsigned char db[16] = {0}, pb[16] = {0}, fb[16] = {0}, gb[16] = {0};
                size_t nb = x.bytes < 16 ? x.bytes : 16;
                memcpy(db, dense[t].data(), nb); memcpy(pb, d.post.data() + this_off, nb);
                FILE* pf = std::fopen((d.dir + "/post.bin").c_str(), "rb");
                FILE* rf = std::fopen((d.dir + "/pre.bin").c_str(), "rb");
                if (pf) { if (fseek(pf, (long)this_off, SEEK_SET) == 0) { size_t r = fread(fb, 1, nb, pf); (void)r; } std::fclose(pf); }
                if (rf) { if (fseek(rf, (long)this_off, SEEK_SET) == 0) { size_t r = fread(gb, 1, nb, rf); (void)r; } std::fclose(rf); }
                std::printf("  DBG op%zu off=%zu postsize=%zu presize=%zu\n", t, this_off, d.post.size(), d.pre.size());
                std::printf("    dense: "); for (int q = 0; q < 16; ++q) std::printf("%02x", db[q]);
                std::printf("\n    mempost: "); for (int q = 0; q < 16; ++q) std::printf("%02x", pb[q]);
                std::printf("\n    filepost: "); for (int q = 0; q < 16; ++q) std::printf("%02x", fb[q]);
                std::printf("\n    filepre: "); for (int q = 0; q < 16; ++q) std::printf("%02x", gb[q]);
                std::printf("\n");
            }
            /* locate differing elements through the stride map (skip pure-gap bytes:
             * ref cannot write them, so a diff there means the CARD wrote outside its
             * logical shape -- itself worth seeing) */
            size_t e = esz(x.dt), bad = 0, shown = 0;
            for (size_t b = 0; b + e <= x.bytes; b += e)
                if (std::memcmp(dense[t].data() + b, d.post.data() + this_off + b, e) != 0) {
                    if (shown < 4) {
                        double va = 0, vb = 0;
                        if (x.dt == RAD_F32) { float fa, fb; std::memcpy(&fa, d.post.data() + this_off + b, 4); std::memcpy(&fb, dense[t].data() + b, 4); va = fa; vb = fb; }
                        else if (x.dt == RAD_BF16) {
                            uint32_t ua = (uint32_t)((uint16_t)d.post[this_off+b+0] | ((uint16_t)d.post[this_off+b+1] << 8)) << 16;
                            uint32_t ub = (uint32_t)((uint16_t)dense[t][b+0] | ((uint16_t)dense[t][b+1] << 8)) << 16;
                            float f; std::memcpy(&f, &ua, 4); va = f; std::memcpy(&f, &ub, 4); vb = f;
                        }
                        std::printf("  MISMATCH operand %zu byte %zu: card=%.9g ref=%.9g\n", t, b, va, vb);
                        ++shown;
                    }
                    ++bad;
                }
            std::printf("  operand %zu: %zu byte-group(s) differ (%s)\n", t, bad,
                        contig_at(x) ? "contiguous" : "span incl. gaps");
            ++fails;
        }
        total += fails + unver;
        std::printf("  VERDICT: %s%s\n",
                    fails ? "DIVERGENT" : (unver ? "EXACT where verifiable" : "EXACT"),
                    unver ? "  (some operands REF NO-WRITE)" : "");
        if (!fails && !unver) std::printf("  ALL EQUAL -- ref matches the card on the engine's own inputs\n");
    }
    std::printf("\n%s: %d issue group(s)\n", total ? "NEEDS LOOK" : "ALL EQUAL", total);
    return total ? 1 : 0;
}
