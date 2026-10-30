/* q61b_embedq_probe -- dedicated battery + microbench for libr3d's `embed_lookup_q` device
 * arm (Window 77, Q61b). Battery law: no committed .rkb pack dispatches embed_lookup_q
 * (stock kernels.rkb carries 10 embed_lookup rows and zero _q; the mxfp4/27B/fp8 packs
 * likewise none), so this is the "bands the battery is blind to" harness: a dedicated
 * oracle + A/B over exactly the shapes the 27B fp8-embd line serves (248320 x 5120), at
 * the card.
 *
 * The oracle is libref: ref_gemm.cpp's embed_gather / ref_embed_lookup_q loop (branch
 * w76-q61-fp8-embd @ 8b92197) ported verbatim (Apache-2.0, (c) radiance authors), reading
 * and storing through the very ABI helpers libref uses -- rad_fp8e4m3_to_f32,
 * rad_bf16_to_f32, rad_f32_to_bf16 (rad_plugin.h). Cell-wise bit-equality against it is
 * the Q62 claim under test.
 *
 * Modes:  probe   <libr3d.so>                -- registration audit + edge probe, seconds
 *         battery <libr3d.so>                -- full suite incl. exhaustive 248320x5120
 *         bench   <libr3d.so> <libref.so|->  -- interleaved device-vs-host microbench
 *
 * Run at the card with ROCR_VISIBLE_DEVICES pinned to the gfx1100; the binary asserts its
 * device name (a battery on a gfx1201 is a phantom -- the gate.sh law).
 */
#include <hip/hip_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rad_abi.h"
#include "rad_plugin.h"

extern "C" int rad_kernel_count(void);
extern "C" const RadKernelInfo* rad_kernel_at(int);
extern "C" int rad_kernel_schema_count(void);
extern "C" const RadOpSchema* rad_kernel_schema_at(int);
static int (*p_kernel_count)(void);
static const RadKernelInfo* (*p_kernel_at)(int);
static int (*p_schema_count)(void);
static const RadOpSchema* (*p_schema_at)(int);

#define CHECK_HIP(e)                                                        \
    do {                                                                    \
        hipError_t e_ = (e);                                                \
        if (e_ != hipSuccess) {                                             \
            fprintf(stderr, "HIP %s @%d: %s\n", #e, __LINE__,               \
                    hipGetErrorString(e_));                                 \
            exit(3);                                                        \
        }                                                                   \
    } while (0)

typedef int (*launch_fn)(const RadArgs*, RadStream);
static launch_fn g_entry = nullptr;
static int g_fail = 0, g_pass = 0;
static bool g_nan_exact = false;   /* set by the NaN audit: device == host bit-for-bit
                                    * on NaN products (then raw patterns everywhere) */

static void load_entry(const char* path) {
    void* dl = dlopen(path, RTLD_NOW);
    if (!dl) { fprintf(stderr, "dlopen %s: %s\n", path, dlerror()); exit(3); }
    g_entry = (launch_fn)dlsym(dl, "r3d_rad_embed_lookup_q");
    if (!g_entry) { fprintf(stderr, "no r3d_rad_embed_lookup_q in %s\n", path); exit(3); }
    p_kernel_count = (int (*)(void))dlsym(dl, "rad_kernel_count");
    p_kernel_at = (const RadKernelInfo * (*)(int))dlsym(dl, "rad_kernel_at");
    p_schema_count = (int (*)(void))dlsym(dl, "rad_kernel_schema_count");
    p_schema_at = (const RadOpSchema * (*)(int))dlsym(dl, "rad_kernel_schema_at");
}

static void require_gfx1100() {
    int dev = 0;
    hipDeviceProp_t p{};
    CHECK_HIP(hipGetDevice(&dev));
    CHECK_HIP(hipGetDeviceProperties(&p, dev));
    printf("device: %s\n", p.gcnArchName);
    if (strncmp(p.gcnArchName, "gfx11", 5) != 0) {
        fprintf(stderr,
                "FATAL: embed_lookup_q battery must run on the gfx1100 alone, got %s "
                "(ROCR_VISIBLE_DEVICES wrong?) -- refusing to report.\n",
                p.gcnArchName);
        exit(4);
    }
}

/* splitmix64 -- deterministic byte pattern; the fill kernel and the host oracle walk the
 * same function of the byte index, so the exhaustive passes never stage the 1.2 GiB codes
 * plane in host RAM (the box has ~2 GiB free; flashnext holds the rest). */
__host__ __device__ static inline uint64_t sm64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
/* E4M3 has exactly two NaN bit patterns (0x7F, 0xFF). The nan_free spelling remaps ONLY
 * those, onto their finite neighbours; the NaN class itself is audited by nan_audit below,
 * whose result decides whether the big passes use raw or remapped patterns. */
__host__ __device__ static inline uint8_t code_pat(uint64_t i, int nan_free) {
    uint8_t c = (uint8_t)(sm64(i) >> 29);
    if (nan_free && c == 0x7Fu) c = 0x7Eu;
    if (nan_free && c == 0xFFu) c = 0xFEu;
    return c;
}
__global__ void fill_pat(uint8_t* p, size_t n, int nan_free) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = code_pat(i, nan_free);
}

/* ------------------------------------------------------------- the libref oracle */
struct Oracle {
    const void* plane = nullptr;   /* scale operand bytes (bf16 or f32) */
    bool plane_f32 = false;
    bool block = false;
    int64_t sc_cols = 0, blk_r = 1, blk_c = 1;
    float factor = 1.0f;           /* the single/absent scale factor */
    int64_t table_rows = 0;
};
static inline float o_scale(const Oracle& o, int64_t id, int64_t i) {
    if (!o.block) return o.factor;
    const int64_t off = (id / o.blk_r) * o.sc_cols + i / o.blk_c;
    return o.plane_f32 ? ((const float*)o.plane)[off]
                       : rad_bf16_to_f32(((const uint16_t*)o.plane)[off]);
}
static inline uint16_t o_cell(const Oracle& o, float v, int64_t id, int64_t i) {
    if (id < 0 || id >= o.table_rows) return 0;               /* ref's have_row */
    return rad_f32_to_bf16(v * o_scale(o, id, i));
}

/* ------------------------------------------------------------- launch plumbing */
struct Ctx {
    void* codes = nullptr;
    void* scale = nullptr;
    void* tok = nullptr;
    uint16_t* x = nullptr;
    uint32_t codes_dtype = RAD_F8E4M3, scale_dtype = RAD_BF16;
    int64_t n_vocab = 0, n_embd = 0, wte_pitch = 0, x_pitch = 0;
    int64_t sc_rows = 1, sc_cols = 1;
    std::vector<int32_t> htok;
};

static int launch(Ctx& c, int64_t M, int tok64, int64_t voff) {
    RadTensor t[5]{};
    t[0].data = c.tok;
    t[0].dtype = tok64 ? RAD_I64 : RAD_I32;
    t[0].rank = 1; t[0].shape[0] = (int)M; t[0].stride[0] = 1;
    t[1].data = c.codes; t[1].dtype = c.codes_dtype; t[1].rank = 2;
    t[1].shape[0] = (int)c.n_vocab; t[1].shape[1] = (int)c.n_embd;
    t[1].stride[0] = c.wte_pitch; t[1].stride[1] = 1;
    if (c.scale) {
        t[2].data = c.scale; t[2].dtype = c.scale_dtype; t[2].rank = 2;
        t[2].shape[0] = (int)c.sc_rows; t[2].shape[1] = (int)c.sc_cols;
        t[2].stride[0] = c.sc_cols; t[2].stride[1] = 1;
    }
    t[3].data = c.x; t[3].dtype = RAD_BF16; t[3].rank = 2;
    t[3].shape[0] = (int)M; t[3].shape[1] = (int)c.n_embd;
    t[3].stride[0] = c.x_pitch; t[3].stride[1] = 1;
    /* operand 4 `ahead`: absent (data null) -- the in-place reader must ignore it. */
    RadParam pv[] = {
        RAD_INT("M", M), RAD_INT("n_embd", c.n_embd), RAD_INT("n_vocab", c.n_vocab),
        RAD_STR("dtype", "bf16"), RAD_INT("vocab_offset", voff) };
    RadArgs a{};
    a.t = t; a.n_t = 5; a.p = pv; a.n_p = 5;
    const int rc = g_entry(&a, (RadStream)0);
    if (rc == RAD_OK) CHECK_HIP(hipDeviceSynchronize());
    return rc;
}

/* Compare x against the oracle, chunk by chunk. codes: explicit host mirror (small
 * cases) or the splitmix pattern (large passes). */
struct CmpReq {
    Ctx* c; int64_t M; int64_t voff; Oracle o;
    const void* hcodes = nullptr;   /* u8 or u16 mirror; null => pattern generator */
    int max_show = 8;
};
static int64_t g_nan_class = 0;     /* cells where both sides are NaN but bits differ */
static int compare_eq(CmpReq& r) {
    const int64_t K = r.c->n_embd, P = r.c->x_pitch;
    const int64_t CH = 4096;
    std::vector<uint16_t> hx((size_t)CH * P);
    std::vector<uint8_t> hc;          /* bytes: an fp8 codes mirror, never a u16 view */
    if (r.hcodes && r.c->codes_dtype == RAD_F8E4M3) hc.resize((size_t)CH * K);
    int bad = 0, shown = 0;
    for (int64_t m0 = 0; m0 < r.M; m0 += CH) {
        const int64_t mr = r.M - m0 < CH ? r.M - m0 : CH;
        if (r.hcodes) {
            if (r.c->codes_dtype == RAD_F8E4M3)
                for (int64_t m = 0; m < mr; ++m) {
                    int64_t id = (int64_t)r.c->htok[m0 + m] - r.voff;
                    if (id < 0) id = 0;
                    if (id >= r.c->n_vocab) id = r.c->n_vocab - 1;
                    memcpy(hc.data() + m * K, (const uint8_t*)r.hcodes + id * r.c->wte_pitch, K);
                }
        }
        CHECK_HIP(hipMemcpy(hx.data(), r.c->x + m0 * P, mr * P * 2, hipMemcpyDeviceToHost));
        for (int64_t m = 0; m < mr; ++m) {
            const int64_t id = (int64_t)r.c->htok[m0 + m] - r.voff;
            for (int64_t i = 0; i < K; ++i) {
                float v;
                if (r.c->codes_dtype == RAD_F8E4M3) {
                    const uint8_t by = r.hcodes ? hc[m * K + i]
                                                : code_pat((uint64_t)id * K + i,
                                                           /*nan_free*/ !g_nan_exact);
                    v = rad_fp8e4m3_to_f32(by);
                } else {
                    int64_t idc = id < 0 ? 0 : (id >= r.c->n_vocab ? r.c->n_vocab - 1 : id);
                    const uint16_t w =
                        ((const uint16_t*)r.hcodes)[idc * r.c->wte_pitch + i];
                    v = rad_bf16_to_f32(w);
                }
                const uint16_t want = o_cell(r.o, v, id, i);
                const uint16_t got = hx[m * P + i];
                if (got == want) continue;
                const bool gnan = (got & 0x7F80u) == 0x7F80u && (got & 0x007Fu) != 0;
                const bool wnan = (want & 0x7F80u) == 0x7F80u && (want & 0x007Fu) != 0;
                if (gnan && wnan) { ++g_nan_class; continue; }  /* IEEE-unspecified */
                if (shown++ < r.max_show)
                    printf("   m=%lld id=%lld i=%lld got %04x want %04x\n",
                           (long long)(m0 + m), (long long)id, (long long)i, got, want);
                ++bad;
            }
        }
    }
    return bad;
}

static void report(const char* what, int rc, int bad, long long cells) {
    const bool ok = rc == RAD_OK && bad == 0;
    printf("%-4s %-56s rc=%d cells=%lld bad=%d\n", ok ? "PASS" : "FAIL", what, rc, cells, bad);
    if (ok) ++g_pass; else ++g_fail;
}

/* ------------------------------------------------------------- scale planes */
static std::vector<uint16_t> plane_random(int64_t r, int64_t cc, uint64_t seed,
                                          bool specials) {
    std::vector<uint16_t> s((size_t)r * cc);
    for (int64_t i = 0; i < r; ++i)
        for (int64_t j = 0; j < cc; ++j) {
            const uint64_t h = sm64(seed + (uint64_t)i * 1315423911u + (uint64_t)j * 2654435761u);
            uint16_t u = (uint16_t)(h >> 32);
            const uint16_t e = (uint16_t)(0x70 + ((h >> 16) % 21)); /* exp in [1/1024,1024] */
            u = (uint16_t)((u & 0x807Fu) | (e << 7));
            if (specials && ((h >> 24) & 0x3F) == 0)
                switch ((h >> 30) & 3) {
                    case 0: u = 0x0000; break;               /* +0 */
                    case 1: u = 0x8000; break;               /* -0 */
                    case 2: u = 0x7F80; break;               /* Inf */
                    default: u = 0x7FC0; break;              /* NaN */
                }
            s[(size_t)i * cc + j] = u;
        }
    return s;
}

/* ------------------------------------------------------------- one case */
struct Case {
    std::string tag;
    int64_t M = 64, n_vocab = 384, n_embd = 320;
    bool codes_bf16 = false;
    int scale_mode = 3;          /* 0 absent | 1 single bf16 | 2 single f32 | 3 plane bf16 | 4 plane f32 */
    int sc_r = 0, sc_c = 0;      /* plane dims; default ceil(n_vocab/128) x ceil(n_embd/128) */
    bool tok64 = false;
    int64_t voff = 0;
    int64_t x_pitch_extra = 0, wte_pitch_extra = 0;
    int x_base_off = 0;
    bool specials = false;
    int exp_rc = RAD_OK;         /* a case the kernel must REFUSE: the row passes when rc matches */
    std::vector<int32_t> ids;    /* explicit GLOBAL ids (may be OOB); empty => PRNG in-range */
};

static void run_case(Case& cs) {
    const int64_t sc_r = cs.sc_r ? cs.sc_r : (cs.n_vocab + 127) / 128;
    const int64_t sc_c = cs.sc_c ? cs.sc_c : (cs.n_embd + 127) / 128;
    Ctx c{};
    c.n_vocab = cs.n_vocab; c.n_embd = cs.n_embd;
    c.wte_pitch = cs.n_embd + cs.wte_pitch_extra;
    c.x_pitch = cs.n_embd + cs.x_pitch_extra;
    c.codes_dtype = cs.codes_bf16 ? RAD_BF16 : RAD_F8E4M3;
    c.scale_dtype = (cs.scale_mode == 2 || cs.scale_mode == 4) ? RAD_F32 : RAD_BF16;
    c.sc_rows = cs.scale_mode >= 3 ? sc_r : 1;
    c.sc_cols = cs.scale_mode >= 3 ? sc_c : 1;

    std::vector<uint8_t> hcodes8;
    std::vector<uint16_t> hcodes16;
    const int nan_free = g_nan_exact ? 0 : 1;
    if (cs.codes_bf16) {
        hcodes16.assign((size_t)cs.n_vocab * c.wte_pitch, 0);
        for (int64_t id = 0; id < cs.n_vocab; ++id)
            for (int64_t i = 0; i < cs.n_embd; ++i) {
                uint16_t v = (uint16_t)(sm64(0xA50000ULL + (uint64_t)id * 2654435761u + i) >> 32);
                if ((v & 0x7F80u) == 0x7F80u) v &= 0x7F7Fu;    /* finite only */
                hcodes16[(size_t)id * c.wte_pitch + i] = v;
            }
        CHECK_HIP(hipMalloc(&c.codes, hcodes16.size() * 2));
        CHECK_HIP(hipMemcpy(c.codes, hcodes16.data(), hcodes16.size() * 2, hipMemcpyHostToDevice));
    } else {
        const size_t n = (size_t)cs.n_vocab * c.wte_pitch;
        CHECK_HIP(hipMalloc(&c.codes, n));
        if (n <= (size_t)16 << 20) {                            /* small: keep host mirror */
            hcodes8.resize(n);
            for (size_t i = 0; i < n; ++i) hcodes8[i] = code_pat(i, nan_free);
            CHECK_HIP(hipMemcpy(c.codes, hcodes8.data(), n, hipMemcpyHostToDevice));
        } else {
            fill_pat<<<(unsigned)((n + 255) / 256), 256>>>((uint8_t*)c.codes, n, nan_free);
            CHECK_HIP(hipGetLastError());
        }
    }
    std::vector<uint16_t> hplane;
    if (cs.scale_mode) {
        const size_t np = (size_t)c.sc_rows * c.sc_cols;
        CHECK_HIP(hipMalloc(&c.scale, np * (c.scale_dtype == RAD_F32 ? 4 : 2)));
        if (cs.scale_mode >= 3) {
            hplane = plane_random(c.sc_rows, c.sc_cols, 0x5EED00ULL, cs.specials);
            if (c.scale_dtype == RAD_F32) {
                std::vector<float> f(np);
                for (size_t i = 0; i < np; ++i) f[i] = rad_bf16_to_f32(hplane[i]);
                CHECK_HIP(hipMemcpy(c.scale, f.data(), np * 4, hipMemcpyHostToDevice));
            } else
                CHECK_HIP(hipMemcpy(c.scale, hplane.data(), np * 2, hipMemcpyHostToDevice));
        } else {
            const uint16_t one16 = 0x3F80; const float one32 = 1.0f;
            CHECK_HIP(hipMemcpy(c.scale, c.scale_dtype == RAD_F32 ? (void*)&one32 : (void*)&one16,
                                c.scale_dtype == RAD_F32 ? 4 : 2, hipMemcpyHostToDevice));
        }
    }
    const int64_t M = cs.M;
    c.htok.assign(M, 0);
    for (int64_t m = 0; m < M; ++m)
        c.htok[m] = !cs.ids.empty()
                        ? cs.ids[m % cs.ids.size()]
                        : (int32_t)(cs.voff + (int64_t)(sm64(0xE111ULL + m) >> 33) % cs.n_vocab);
    if (cs.tok64) {
        std::vector<int64_t> t6(M);
        for (int64_t m = 0; m < M; ++m) t6[m] = c.htok[m];
        CHECK_HIP(hipMalloc(&c.tok, M * 8));
        CHECK_HIP(hipMemcpy(c.tok, t6.data(), M * 8, hipMemcpyHostToDevice));
    } else {
        CHECK_HIP(hipMalloc(&c.tok, M * 4));
        CHECK_HIP(hipMemcpy(c.tok, c.htok.data(), M * 4, hipMemcpyHostToDevice));
    }
    uint16_t* xraw = nullptr;
    CHECK_HIP(hipMalloc((void**)&xraw, ((size_t)(M - 1) * c.x_pitch + cs.n_embd + 8) * 2));
    c.x = xraw + cs.x_base_off;
    CHECK_HIP(hipMemset(c.x, 0x5A, ((size_t)(M - 1) * c.x_pitch + cs.n_embd) * 2));

    const int rc = launch(c, M, cs.tok64, cs.voff);
    if (rc == RAD_OK) {
        Oracle o{};
        o.table_rows = cs.n_vocab;
        o.block = cs.scale_mode >= 3;
        o.plane_f32 = c.scale_dtype == RAD_F32 && o.block;
        if (o.block) {
            std::vector<uint16_t> got((size_t)c.sc_rows * c.sc_cols);
            if (o.plane_f32) {
                std::vector<float> f(got.size());
                CHECK_HIP(hipMemcpy(f.data(), c.scale, f.size() * 4, hipMemcpyDeviceToHost));
                o.plane = malloc(f.size() * 4);
                memcpy((void*)o.plane, f.data(), f.size() * 4);
            } else {
                CHECK_HIP(hipMemcpy(got.data(), c.scale, got.size() * 2, hipMemcpyDeviceToHost));
                o.plane = malloc(got.size() * 2);
                memcpy((void*)o.plane, got.data(), got.size() * 2);
            }
            o.sc_cols = c.sc_cols;
            o.blk_r = (cs.n_vocab + c.sc_rows - 1) / c.sc_rows;
            o.blk_c = (cs.n_embd + c.sc_cols - 1) / c.sc_cols;
        } else if (cs.scale_mode) {
            o.factor = 1.0f;                                    /* the seeded single scale */
        }
        CmpReq r{&c, M, cs.voff, o,
                 c.codes_dtype == RAD_F8E4M3
                     ? (const void*)(hcodes8.empty() ? nullptr : hcodes8.data())
                     : (const void*)hcodes16.data()};
        const int bad = compare_eq(r);
        if (o.plane) free((void*)o.plane);
        char tag[192];
        snprintf(tag, sizeof tag, "%s V=%s M=%lld N=%lldxK=%lld", cs.tag.c_str(),
                 (cs.n_embd % 4 == 0 && !cs.x_pitch_extra && !cs.wte_pitch_extra &&
                  !cs.x_base_off && (!o.block || o.blk_c % 4 == 0)) ? "4" : "1",
                 (long long)M, (long long)cs.n_vocab, (long long)cs.n_embd);
        report(tag, rc, bad, (long long)M * cs.n_embd);
    } else if (rc == cs.exp_rc) {
        printf("PASS %-56s rc=%d (refused as expected)\n", cs.tag.c_str(), rc);
        ++g_pass;
    } else {
        report(cs.tag.c_str(), rc, -1, 0);
    }
    if (c.x != xraw) c.x = xraw;
    hipFree(xraw);
    hipFree(c.tok);
    if (c.scale) hipFree(c.scale);
    hipFree(c.codes);
}

/* ------------------------------------------------------------- exhaustive pass */
static void exhaustive(int64_t NV, int64_t K, int seed_specials) {
    const int64_t SR = (NV + 127) / 128, SC = (K + 127) / 128;
    Ctx c{};
    c.n_vocab = NV; c.n_embd = K; c.wte_pitch = K; c.x_pitch = K;
    c.codes_dtype = RAD_F8E4M3; c.scale_dtype = RAD_BF16;
    c.sc_rows = SR; c.sc_cols = SC;
    CHECK_HIP(hipMalloc(&c.codes, (size_t)NV * K));
    fill_pat<<<(unsigned)(((size_t)NV * K + 255) / 256), 256>>>((uint8_t*)c.codes,
                                                                (size_t)NV * K,
                                                                g_nan_exact ? 0 : 1);
    CHECK_HIP(hipGetLastError());
    auto plane = seed_specials ? plane_random(SR, SC, 0xC0FFEEULL, true)
                               : std::vector<uint16_t>((size_t)SR * SC, 0x3F80);
    CHECK_HIP(hipMalloc(&c.scale, plane.size() * 2));
    CHECK_HIP(hipMemcpy(c.scale, plane.data(), plane.size() * 2, hipMemcpyHostToDevice));
    const int64_t M = NV;                                       /* every row: exhaustive */
    std::vector<int32_t> ids(M);
    for (int64_t m = 0; m < M; ++m) ids[m] = (int32_t)m;
    c.htok = ids;
    CHECK_HIP(hipMalloc((void**)&c.tok, M * 4));
    CHECK_HIP(hipMemcpy(c.tok, ids.data(), M * 4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMalloc((void**)&c.x, (size_t)M * K * 2));
    CHECK_HIP(hipMemset(c.x, 0x5A, (size_t)M * K * 2));
    const int rc = launch(c, M, 0, 0);
    if (rc == RAD_OK) {
        Oracle o{};
        o.table_rows = NV; o.block = true; o.sc_cols = SC;
        o.blk_r = (NV + SR - 1) / SR; o.blk_c = (K + SC - 1) / SC;
        o.plane = plane.data(); o.plane_f32 = false;
        CmpReq r{&c, M, 0, o, nullptr, 12};
        const int bad = compare_eq(r);
        char tag[128];
        snprintf(tag, sizeof tag, "exhaustive %lldx%lld scale=%s", (long long)NV,
                 (long long)K, seed_specials ? "random+specials" : "all-ones");
        report(tag, rc, bad, (long long)M * K);
    } else
        report("exhaustive", rc, -1, 0);
    hipFree(c.x); hipFree(c.tok); hipFree(c.scale); hipFree(c.codes);
}

/* ------------------------------------------------------------- NaN audit */
static void nan_audit() {
    /* one row, K=256, codes = all 256 E4M3 patterns; scales: +1, -1, 0, Inf, NaN, 448 */
    static const uint16_t scales[] = {0x3F80, 0xBF80, 0x0000, 0x7F80, 0x7FC0, 0x43E0};
    const char* names[] = {"+1.0", "-1.0", "+0.0", "+Inf", "NaN", "448.0"};
    for (int s = 0; s < 6; ++s) {
        Ctx c{};
        const int64_t K = 256;
        c.n_vocab = 1; c.n_embd = K; c.wte_pitch = K; c.x_pitch = K;
        c.codes_dtype = RAD_F8E4M3; c.scale_dtype = RAD_BF16;
        uint8_t codes[256];
        for (int i = 0; i < 256; ++i) codes[i] = (uint8_t)i;
        const int32_t tok0 = 0;
        CHECK_HIP(hipMalloc(&c.codes, 256));
        CHECK_HIP(hipMemcpy(c.codes, codes, 256, hipMemcpyHostToDevice));
        CHECK_HIP(hipMalloc(&c.tok, 4));
        CHECK_HIP(hipMemcpy(c.tok, &tok0, 4, hipMemcpyHostToDevice));
        CHECK_HIP(hipMalloc((void**)&c.x, K * 2));
        CHECK_HIP(hipMemset(c.x, 0x5A, K * 2));
        /* the scale under test must reach the kernel: a one-entry bf16 operand (ref: an
         * operand with a single entry is the whole-table scale). Without it the audit
         * measured the unscaled gather against a scaled oracle. */
        CHECK_HIP(hipMalloc(&c.scale, 2));
        CHECK_HIP(hipMemcpy(c.scale, &scales[s], 2, hipMemcpyHostToDevice));
        c.htok = {0};
        const int rc = launch(c, 1, 0, 0);
        if (rc == RAD_OK) {
            std::vector<uint16_t> got(K);
            CHECK_HIP(hipMemcpy(got.data(), c.x, K * 2, hipMemcpyDeviceToHost));
            int bad = 0, ncls = 0;
            for (int i = 0; i < 256; ++i) {
                const float v = rad_fp8e4m3_to_f32(codes[i]);
                const float sv = rad_bf16_to_f32(scales[s]);
                const uint16_t want = rad_f32_to_bf16(v * sv);
                if (got[i] == want) continue;
                const bool gnan = (got[i] & 0x7F80u) == 0x7F80u && (got[i] & 0x7Fu);
                const bool wnan = (want & 0x7F80u) == 0x7F80u && (want & 0x7Fu);
                if (gnan && wnan) { ++ncls; continue; }
                ++bad;
                if (bad < 4)
                    printf("   code %02x scale %s: got %04x want %04x\n", codes[i],
                           names[s], got[i], want);
            }
            char tag[128];
            snprintf(tag, sizeof tag, "nan-audit all-256 codes x scale %s", names[s]);
            report(tag, rc, bad, 256);
            if (ncls) printf("     (%lld NaN-class bit-divergences, IEEE-unspecified)\n",
                             (long long)ncls);
        } else
            report("nan-audit launch", rc, -1, 0);
        hipFree(c.x); hipFree(c.tok); hipFree(c.scale); hipFree(c.codes);
    }
}

/* ------------------------------------------------------------- registration audit */
static void audit_registration() {
    int row_seen = 0, schema_seen = 0;
    if (!p_kernel_count || !p_kernel_at || !p_schema_count || !p_schema_at) {
        printf("FAIL registration: registry exports missing from plugin\n");
        ++g_fail;
        return;
    }
    for (int i = 0; i < p_kernel_count(); ++i) {
        const RadKernelInfo* k = p_kernel_at(i);
        if (k && !strcmp(k->op, "embed_lookup_q")) {
            ++row_seen;
            printf("row: %s domain=%d priority=%d\n", k->name, k->domain, k->priority);
        }
    }
    for (int i = 0; i < p_schema_count(); ++i) {
        const RadOpSchema* s = p_schema_at(i);
        if (s && !strcmp(s->op, "embed_lookup_q")) {
            ++schema_seen;
            static const char* pk[] = {"M", "n_embd", "n_vocab", "dtype", "vocab_offset",
                                       "wscale"};
            int ok = s->n_params == 6;
            for (int j = 0; j < s->n_params && j < 6; ++j)
                if (strcmp(s->params[j].key, pk[j])) ok = 0;
            ok = ok && s->n_operands == 5 && s->operands[2].optional &&
                 s->operands[2].role == RAD_OPD_WEIGHT && s->operands[4].optional &&
                 s->operands[4].role == RAD_OPD_IN;
            printf("schema: params=%d operands=%d %s\n", s->n_params, s->n_operands,
                   ok ? "matches libref's pEmbed/oEmbedQ" : "MISMATCH vs libref");
            if (!ok) ++g_fail;
        }
    }
    printf("%-4s registration: %d row(s), %d schema(s) for embed_lookup_q\n",
           row_seen == 1 && schema_seen == 1 ? "PASS" : "FAIL", row_seen, schema_seen);
    if (row_seen != 1 || schema_seen != 1) ++g_fail;
}

/* ------------------------------------------------------------- refusal matrix */
static void refusals() {
    struct R { const char* what; int want; } ;
    Ctx c{};
    const int64_t NV = 256, K = 128;
    c.n_vocab = NV; c.n_embd = K; c.wte_pitch = K; c.x_pitch = K;
    c.codes_dtype = RAD_F8E4M3; c.scale_dtype = RAD_BF16; c.sc_rows = 2; c.sc_cols = 1;
    CHECK_HIP(hipMalloc(&c.codes, NV * K));
    CHECK_HIP(hipMemset(c.codes, 0x30, NV * K));
    CHECK_HIP(hipMalloc((void**)&c.tok, 4));
    c.htok = {5};
    CHECK_HIP(hipMemcpy(c.tok, c.htok.data(), 4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMalloc((void**)&c.x, K * 2));

    auto launch_with = [&](uint32_t tok_dt, uint32_t wte_dt, uint32_t sc_dt, int sc_rank,
                           int64_t sc_stride, int64_t voff, int n_t, int64_t nn,
                           int64_t kk) -> int {
        RadTensor t[5]{};
        t[0].data = c.tok; t[0].dtype = tok_dt; t[0].rank = 1; t[0].shape[0] = 1;
        t[0].stride[0] = 1;
        t[1].data = c.codes; t[1].dtype = wte_dt; t[1].rank = 2; t[1].shape[0] = (int)NV;
        t[1].shape[1] = (int)K; t[1].stride[0] = K; t[1].stride[1] = 1;
        if (sc_dt) {
            CHECK_HIP(hipMalloc(&c.scale, 64));
            t[2].data = c.scale; t[2].dtype = sc_dt; t[2].rank = sc_rank;
            t[2].shape[0] = 2; t[2].shape[1] = 1; t[2].stride[0] = sc_stride;
            t[2].stride[1] = 1;
        }
        t[3].data = c.x; t[3].dtype = RAD_BF16; t[3].rank = 2; t[3].shape[0] = 1;
        t[3].shape[1] = (int)K; t[3].stride[0] = K; t[3].stride[1] = 1;
        RadParam pv[] = {
            RAD_INT("M", 1), RAD_INT("n_embd", kk), RAD_INT("n_vocab", nn),
            RAD_STR("dtype", "bf16"), RAD_INT("vocab_offset", voff) };
        RadArgs a{}; a.t = t; a.n_t = n_t; a.p = pv; a.n_p = 5;
        const int rc = g_entry(&a, (RadStream)0);
        if (c.scale) { hipFree(c.scale); c.scale = nullptr; }
        return rc;
    };
    auto chk = [&](const char* what, int want, int got) {
        printf("%-4s %-56s rc=%d want=%d\n", got == want ? "PASS" : "FAIL", what, got, want);
        if (got == want) ++g_pass; else ++g_fail;
    };
    chk("x f32 refused", RAD_E_DTYPE, [&]{
        RadTensor t[5]{};
        t[0].data=c.tok; t[0].dtype=RAD_I32; t[0].rank=1; t[0].shape[0]=1; t[0].stride[0]=1;
        t[1].data=c.codes; t[1].dtype=RAD_F8E4M3; t[1].rank=2; t[1].shape[0]=(int)NV;
        t[1].shape[1]=(int)K; t[1].stride[0]=K; t[1].stride[1]=1;
        t[2].data=nullptr; t[2].dtype=RAD_F8E4M3;
        t[3].data=c.x; t[3].dtype=RAD_F32; t[3].rank=2; t[3].shape[0]=1; t[3].shape[1]=(int)K;
        t[3].stride[0]=K; t[3].stride[1]=1;
        RadParam pv[]={RAD_INT("M",1),RAD_INT("n_embd",K),RAD_INT("n_vocab",NV),
                       RAD_STR("dtype","bf16")};
        RadArgs a{}; a.t=t; a.n_t=4; a.p=pv; a.n_p=4;
        return g_entry(&a,(RadStream)0); }());
    chk("scale f16 refused", RAD_E_DTYPE, launch_with(RAD_I32, RAD_F8E4M3, RAD_F16, 2, 1, 0, 5, NV, K));
    chk("bf16 table + plane scale refused", RAD_E_SHAPE,
        launch_with(RAD_I32, RAD_BF16, RAD_BF16, 2, 1, 0, 5, NV, K));
    chk("rank-1 plane scale refused", RAD_E_SHAPE,
        launch_with(RAD_I32, RAD_F8E4M3, RAD_BF16, 1, 1, 0, 5, NV, K));
    chk("non-contig scale refused", RAD_E_SHAPE,
        launch_with(RAD_I32, RAD_F8E4M3, RAD_BF16, 2, 9, 0, 5, NV, K));
    chk("tok f32 refused", RAD_E_DTYPE, launch_with(RAD_F32, RAD_F8E4M3, 0, 2, 1, 0, 5, NV, K));
    chk("vocab_offset < 0 refused", RAD_E_SHAPE,
        launch_with(RAD_I32, RAD_F8E4M3, 0, 2, 1, -1, 5, NV, K));
    chk("n_embd mismatch x refused", RAD_E_SHAPE,
        launch_with(RAD_I32, RAD_F8E4M3, 0, 2, 1, 0, 5, NV, K + 8));
    chk("n_vocab beyond table refused", RAD_E_SHAPE,
        launch_with(RAD_I32, RAD_F8E4M3, 0, 2, 1, 0, 5, NV * 8, K));
    hipFree(c.x); hipFree(c.tok); hipFree(c.codes);
}

/* ------------------------------------------------------------- modes */
static void battery() {
    audit_registration();
    nan_audit();
    g_nan_exact = g_nan_class == 0;
    printf("nan policy: exhaustive patterns %s\n",
           g_nan_exact ? "RAW (NaN products bit-equal)" : "nan-free (NaN class audited separately)");

    /* forms at a 384x320-ish shape: table x scale spelling x tokens x M */
    Case base;
    base.M = 33; base.n_vocab = 384; base.n_embd = 320;
    struct V { Case c; };
    std::vector<Case> v;
    auto push = [&](const char* tag, auto&&... f) {
        Case x = base;
        x.tag = tag;
        (f(x), ...);
        v.push_back(x);
    };
    push("fp8 plane bf16 scale", [](Case& x) {});
    push("fp8 plane f32 scale", [](Case& x) { x.scale_mode = 4; });
    push("fp8 single bf16 scale", [](Case& x) { x.scale_mode = 1; });
    push("fp8 single f32 scale", [](Case& x) { x.scale_mode = 2; });
    push("fp8 no scale (=1)", [](Case& x) { x.scale_mode = 0; });
    push("bf16 table plane scale refused", [](Case& x) {
        x.codes_bf16 = true; x.exp_rc = RAD_E_SHAPE; });  /* the plane is the block-fp8 form:
        * ref's host loop never type-checks the pair, but the row refuses it at resolve --
        * a schema prose that spells the plane as an fp8-table companion and a device arm
        * that never serves the pair must agree, and a refusal is never wrong numbers. */
    push("bf16 table single scale", [](Case& x) { x.codes_bf16 = true; x.scale_mode = 1; });
    push("bf16 table no scale", [](Case& x) { x.codes_bf16 = true; x.scale_mode = 0; });
    push("plane scale specials (0/-0/Inf/NaN tiles)", [](Case& x) { x.specials = true; });
    push("i64 tokens", [](Case& x) { x.tok64 = true; });
    push("M=1 decode", [](Case& x) { x.M = 1; });
    push("M=2", [](Case& x) { x.M = 2; });
    push("M=4", [](Case& x) { x.M = 4; });
    push("M=257 (grid tail)", [](Case& x) { x.M = 257; });
    push("M=1024 (2 waves of quads)", [](Case& x) { x.M = 1024; });
    /* tile-boundary ids at real tile size: rows 127/128/129 and the last row */
    push("edge ids 0,1,127,128,129,255,256,383 (tile rows)", [](Case& x) {
        x.M = 8; x.ids = {0, 1, 127, 128, 129, 255, 256, 383}; });
    push("edge ids OOB (-1,-500,384,99999) => zero rows", [](Case& x) {
        x.M = 4; x.ids = {-1, -500, 384, 99999}; });
    push("vocab shard voff=100 nv=256 glob ids", [](Case& x) {
        x.n_vocab = 256; x.voff = 100; x.M = 6;
        x.ids = {99, 100, 355, 356, 0, -1}; });
    /* V=1 arms: pitch and base alignment */
    push("V=1 via x_pitch 322", [](Case& x) { x.x_pitch_extra = 2; });
    push("V=1 via odd x base", [](Case& x) { x.x_base_off = 1; });
    push("V=1 via wte_pitch 322", [](Case& x) { x.wte_pitch_extra = 2; });
    push("V=1 via n_embd 318", [](Case& x) { x.n_embd = 318; });
    push("blk_c=40 multiple-of-4 (stays V4, sc_c=8)", [](Case& x) {
        x.n_embd = 320; x.sc_c = 8; });
    push("blk_c%4!=0 forces V=1 block (sc_c=16)", [](Case& x) {
        x.n_embd = 200; x.sc_c = 16; });  /* blk_c=ceil(200/16)=13 */
    push("K=3 tile-boundary cols 0..2", [](Case& x) {
        x.n_embd = 3; x.scale_mode = 1; x.M = 8; });
    push("K=129 (tiles straddle end)", [](Case& x) {
        x.n_embd = 129; x.sc_c = 2; });
    for (auto& cs : v) run_case(cs);

    refusals();

    /* the real shapes: two exhaustive passes over the whole 248320x5120 table */
    exhaustive(248320, 5120, 0);
    exhaustive(248320, 5120, 1);
    /* a prefill-sized gather at the real shapes, compared cell by cell */
    Case pre;
    pre.tag = "prefill M=8192 at 248320x5120 plane";
    pre.M = 8192; pre.n_vocab = 248320; pre.n_embd = 5120;
    run_case(pre);
}

/* ------------------------------------------------------------- bench */
static double now_s() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static void bench(const char* libref_path) {
    /* host fallback = libref's own row (the engine's current fallback path verbatim) */
    launch_fn host_ref = nullptr;
    if (strcmp(libref_path, "-") != 0) {
        void* dl = dlopen(libref_path, RTLD_NOW);
        if (!dl) { fprintf(stderr, "dlopen libref: %s\n", dlerror()); exit(3); }
        auto kc = (int (*)(void))dlsym(dl, "rad_kernel_count");
        auto ka = (const RadKernelInfo* (*)(int))dlsym(dl, "rad_kernel_at");
        if (!kc || !ka) { fprintf(stderr, "libref exports missing\n"); exit(3); }
        for (int i = 0; i < kc(); ++i) {
            const RadKernelInfo* k = ka(i);
            if (k && !strcmp(k->op, "embed_lookup_q")) { host_ref = k->launch;
                printf("host row: %s domain=%d\n", k->name, k->domain); }
        }
        if (!host_ref) { fprintf(stderr, "libref has no embed_lookup_q row\n"); exit(3); }
    }

    const int64_t NV = 248320, K = 5120;   /* the model's real shapes */
    const int64_t SR = (NV + 127) / 128, SC = (K + 127) / 128;

    uint8_t* codes = nullptr;
    uint16_t* plane = nullptr;
    uint16_t* xbig = nullptr;              /* x_pitch K, M up to 8192 */
    int32_t* tokfull = nullptr;            /* 0..NV-1 (full sweep + cold legs) */
    int32_t* tokrand = nullptr;            /* random ids in [0, NV) */
    int32_t* tokfit = nullptr;             /* ids cycling over 1024 distinct rows */
    CHECK_HIP(hipMalloc(&codes, (size_t)NV * K));
    fill_pat<<<(unsigned)(((size_t)NV * K + 255) / 256), 256>>>((uint8_t*)codes, (size_t)NV * K, 1);
    auto hp = plane_random(SR, SC, 0xBEEF00ULL, false);
    CHECK_HIP(hipMalloc(&plane, hp.size() * 2));
    CHECK_HIP(hipMemcpy(plane, hp.data(), hp.size() * 2, hipMemcpyHostToDevice));
    std::vector<int32_t> ids(NV);
    for (int64_t m = 0; m < NV; ++m) ids[m] = (int32_t)m;
    CHECK_HIP(hipMalloc((void**)&tokfull, NV * 4));
    CHECK_HIP(hipMemcpy(tokfull, ids.data(), NV * 4, hipMemcpyHostToDevice));
    std::vector<int32_t> rnd(8192);
    for (size_t i = 0; i < rnd.size(); ++i)
        rnd[i] = (int32_t)((sm64(0xB0ULL + i) >> 33) % (uint64_t)NV);
    CHECK_HIP(hipMalloc((void**)&tokrand, 8192 * 4));
    CHECK_HIP(hipMemcpy(tokrand, rnd.data(), 8192 * 4, hipMemcpyHostToDevice));
    std::vector<int32_t> fit(8192);
    for (int i = 0; i < 8192; ++i) fit[i] = (int32_t)(sm64(0xF170ULL + i) >> 40) % 1024;
    CHECK_HIP(hipMalloc((void**)&tokfit, 8192 * 4));
    CHECK_HIP(hipMemcpy(tokfit, fit.data(), 8192 * 4, hipMemcpyHostToDevice));
    /* the full sweep leg (M = NV) lands every row here: xbig must cover it, not just
     * the 8192-row prefill legs -- an M-sized x smaller than the sweep is a fault. */
    CHECK_HIP(hipMalloc((void**)&xbig, (size_t)NV * K * 2));

    auto device_leg = [&](int32_t* tok, int64_t M) -> double {
        RadTensor t[5]{};
        t[0].data = tok; t[0].dtype = RAD_I32; t[0].rank = 1; t[0].shape[0] = (int)M;
        t[0].stride[0] = 1;
        t[1].data = codes; t[1].dtype = RAD_F8E4M3; t[1].rank = 2; t[1].shape[0] = (int)NV;
        t[1].shape[1] = (int)K; t[1].stride[0] = K; t[1].stride[1] = 1;
        t[2].data = plane; t[2].dtype = RAD_BF16; t[2].rank = 2; t[2].shape[0] = (int)SR;
        t[2].shape[1] = (int)SC; t[2].stride[0] = SC; t[2].stride[1] = 1;
        t[3].data = xbig; t[3].dtype = RAD_BF16; t[3].rank = 2; t[3].shape[0] = (int)M;
        t[3].shape[1] = (int)K; t[3].stride[0] = K; t[3].stride[1] = 1;
        RadParam pv[] = {RAD_INT("M", M), RAD_INT("n_embd", K), RAD_INT("n_vocab", NV),
                         RAD_STR("dtype", "bf16"), RAD_INT("vocab_offset", 0)};
        RadArgs a{}; a.t = t; a.n_t = 5; a.p = pv; a.n_p = 5;
        const double t0 = now_s();
        if (g_entry(&a, (RadStream)0) != RAD_OK) { fprintf(stderr, "device leg failed\n"); exit(3); }
        CHECK_HIP(hipDeviceSynchronize());
        return now_s() - t0;
    };

    /* host table: 8192 rows resident (the box's free RAM bound -- flashnext holds the
     * rest); the engine's host gather reads the FULL table from host DRAM per step, and
     * 8192 rows x 5120 = 40 MiB stays off every x86 LLC, so the rate is the same one the
     * 27B table gets. Documented estimate source for the report. */
    const int64_t HNV = 8192;
    std::vector<uint8_t> hcodes((size_t)HNV * K);
    for (size_t i = 0; i < hcodes.size(); ++i) hcodes[i] = code_pat(i, 1);
    std::vector<uint16_t> hplane((size_t)((HNV + 127) / 128) * SC);
    { auto p = plane_random((HNV + 127) / 128, SC, 0xBEEF00ULL, false);
      hplane = std::move(p); }
    std::vector<uint16_t> hx((size_t)8192 * K);
    std::vector<int32_t> htok(8192);
    for (int i = 0; i < 8192; ++i) htok[i] = (int)(sm64(0xB0 + i) >> 33) % (int32_t)HNV;
    auto host_leg = [&](int64_t M) -> double {
        RadTensor t[5]{};
        t[0].data = htok.data(); t[0].dtype = RAD_I32; t[0].rank = 1; t[0].shape[0] = (int)M;
        t[0].stride[0] = 1;
        t[1].data = hcodes.data(); t[1].dtype = RAD_F8E4M3; t[1].rank = 2;
        t[1].shape[0] = (int)HNV; t[1].shape[1] = (int)K; t[1].stride[0] = K;
        t[1].stride[1] = 1;
        t[2].data = hplane.data(); t[2].dtype = RAD_BF16; t[2].rank = 2;
        t[2].shape[0] = (int)((HNV + 127) / 128); t[2].shape[1] = (int)SC;
        t[2].stride[0] = SC; t[2].stride[1] = 1;
        t[3].data = hx.data(); t[3].dtype = RAD_BF16; t[3].rank = 2; t[3].shape[0] = (int)M;
        t[3].shape[1] = (int)K; t[3].stride[0] = K; t[3].stride[1] = 1;
        RadParam pv[] = {RAD_INT("M", M), RAD_INT("n_embd", K), RAD_INT("n_vocab", HNV),
                         RAD_STR("dtype", "bf16"), RAD_INT("vocab_offset", 0)};
        RadArgs a{}; a.t = t; a.n_t = 5; a.p = pv; a.n_p = 5;
        const double t0 = now_s();
        const int rc = host_ref(&a, (RadStream)0);
        if (rc != RAD_OK) { fprintf(stderr, "host leg rc=%d\n", rc); exit(3); }
        return now_s() - t0;
    };

    /* pre-burn: DVFS ramp per the campaign's leg law (short cold legs are +/-'20%
     * position artifacts without it) */
    {
        std::vector<uint16_t> burn;
        burn.resize((size_t)8192 * K);
        const double t0 = now_s();
        while (now_s() - t0 < 4.0) device_leg(tokrand, 8192);
        if (host_ref) { const double t1 = now_s(); while (now_s() - t1 < 2.0) host_leg(2048); }
        CHECK_HIP(hipDeviceSynchronize());
    }
    printf("pre-burn done (device 4s, host 2s)\n");

    struct LegDef { const char* name; int iters; int cold; };
    const LegDef legs[] = {
        {"decode-M1",   300, 0},   /* one token, sync each: serve decode per-step latency */
        {"decode-M4",   300, 0},
        {"prefill-8192-cache-fit", 30, 0},   /* 1024 distinct rows: 15 MiB L2-resident */
        {"prefill-8192-cold",      12, 1},   /* fresh random rows, 120 MiB moved */
        {"cold-full-sweep",         3, 1},   /* every row: 1.22 GiB codes streamed */
    };
    auto stats = [](std::vector<double>& v) {
        std::sort(v.begin(), v.end());
        return std::pair<double, double>{v.front(), v[v.size() / 2]};
    };

    for (int round = 1; round <= 3; ++round) {
        for (auto& L : legs) {
            /* device leg */
            std::vector<double> d;
            for (int i = 0; i < L.iters; ++i) {
                int32_t* tok = !strcmp(L.name, "cold-full-sweep") ? tokfull
                             : !strcmp(L.name, "prefill-8192-cache-fit") ? tokfit
                             : tokrand;
                const int64_t M = !strcmp(L.name, "cold-full-sweep") ? NV
                                : !strncmp(L.name, "decode", 6)
                                      ? (L.name[8] == '1' ? 1 : 4)
                                      : 8192;
                d.push_back(device_leg(tok, M));
            }
            auto ds = stats(d);
            printf("R%d DEV  %-24s min=%.1fus med=%.1fus", round, L.name, ds.first * 1e6,
                   ds.second * 1e6);
            const int64_t Mmed = !strcmp(L.name, "cold-full-sweep") ? NV
                                 : !strncmp(L.name, "decode", 6)
                                       ? (L.name[8] == '1' ? 1 : 4)
                                       : 8192;
            const double bytes = (double)Mmed * K * 3.0;
            printf(" GB/s=%.1f rows/s=%.0f", bytes / ds.second / 1e9, Mmed / ds.second);
            if (host_ref && strcmp(L.name, "cold-full-sweep") != 0) {
                const int hiters = !strncmp(L.name, "decode", 6) ? 100 : 6;
                std::vector<double> h;
                for (int i = 0; i < hiters; ++i)
                    h.push_back(host_leg(!strncmp(L.name, "decode", 6)
                                             ? (L.name[8] == '1' ? 1 : 4)
                                             : 8192));
                auto hs = stats(h);
                printf("   HOST min=%.1fus med=%.1fus speedup(dev/host)=%.1fx",
                       hs.first * 1e6, hs.second * 1e6, hs.second / ds.second);
            } else if (host_ref) {
                printf("   HOST n/a (no 1.22 GiB host table under the RAM bound)");
            }
            printf("\n");
        }
    }
    printf("NOTE host legs: libref's own embed_lookup_q row (the engine fallback), host-"
           "resident 8192-row table (RAM bound); ratio computed per round, medians.\n");
    printf("NOTE co-resident: spec9g (production) resident on this card all legs.\n");
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s probe|battery|bench <libr3d.so> [libref.so|-]\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    load_entry(argv[2]);
    require_gfx1100();
    if (mode == "probe") {
        audit_registration();
        nan_audit();
        Case cs; cs.tag = "probe fp8 plane 384x320 M=33";
        run_case(cs);
        Case e; e.tag = "probe edge+OOB ids";
        e.M = 12; e.ids = {0, 1, 127, 128, 129, 255, 256, 382, 383, -1, 384, 99999};
        run_case(e);
    } else if (mode == "battery") {
        battery();
    } else if (mode == "bench") {
        if (argc < 4) { fprintf(stderr, "bench needs libref path or -\n"); return 2; }
        bench(argv[3]);
    } else {
        fprintf(stderr, "unknown mode\n");
        return 2;
    }
    printf("TOTAL pass=%d fail=%d nan-class-cells=%lld\n", g_pass, g_fail,
           (long long)g_nan_class);
    return g_fail ? 2 : 0;
}
