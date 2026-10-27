// q53_probe.cpp -- Q53 forensics: fold-vs-plain decode at pinned splits, with
// per-element output dump AND split-KV arena partials (O, m, l) dump.
// Fixture is the bench_attn_decode.hip one verbatim (same seed => same operands
// across processes); the difference is the dump: raw bf16 output + the raw arena
// region (pn*spl*(D+2) f32: pso | psm | psl) written to files for host diff.
// usage: q53_probe dump <outtag> <qh> <kvh> <rows> <ctx>
// env:   LIBR3D_SO, BENCH_HD, R3D_FORCE_SPLITS, R3D_NO_GQA (plain arm)
#include <hip/hip_runtime.h>
#include <dlfcn.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "rad_abi.h"

static void die(const char* m, hipError_t e) {
    if (e == hipSuccess) return;
    fprintf(stderr, "FATAL %s: %s\n", m, hipGetErrorName(e));
    exit(2);
}
static uint32_t xs = 0x2545f491u;                       // same xorshift seed as the bench
static uint32_t rnd32() { xs ^= xs << 13; xs ^= xs >> 17; xs ^= xs << 5; return xs; }
static double rndu(double lo, double hi) { return lo + (hi - lo) * (rnd32() * (1.0 / 4294967296.0)); }
static uint16_t f32_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static void dump(const char* path, const void* p, size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "FATAL open %s\n", path); exit(2); }
    fwrite(p, 1, n, f);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 7 || strcmp(argv[1], "dump")) { printf("usage: dump <tag> <qh> <kvh> <rows> <ctx>\n"); return 1; }
    const char* tag = argv[2];
    const int QH = atoi(argv[3]), KVH = atoi(argv[4]);
    const int rows = atoi(argv[5]), CTX = atoi(argv[6]);
    const int HD = getenv("BENCH_HD") ? atoi(getenv("BENCH_HD")) : 128;
    const int SPL = getenv("R3D_FORCE_SPLITS") ? atoi(getenv("R3D_FORCE_SPLITS")) : 0;
    const int GQA = QH / KVH, BS = 16;
    const int NB = (CTX + BS - 1) / BS;
    const int ROWW = QH * HD, SLOTW = 2 * HD;

    hipDeviceProp_t p;
    die("props", hipGetDeviceProperties(&p, 0));
    if (!strstr(p.gcnArchName, "gfx1100")) { printf("ABORT not-gfx1100\n"); return 3; }

    void* dl = dlopen(getenv("LIBR3D_SO") ? getenv("LIBR3D_SO") : "/rh/kernels/libr3d.so", RTLD_NOW);
    if (!dl) { printf("dlopen: %s\n", dlerror()); return 1; }
    auto entry = (int (*)(const RadArgs*, RadStream))dlsym(dl, "r3d_rad_attn_paged_decode");
    if (!entry) { printf("dlsym: %s\n", dlerror()); return 1; }

    std::vector<uint16_t> hq((size_t)rows * ROWW), hkv((size_t)NB * KVH * BS * SLOTW);
    const bool exact = getenv("Q53_EXACT") != nullptr;   // power-of-2 operands: every QK dot is
                                                          // an exact f32 integer-scaled sum, so
                                                          // any surviving diff is structural,
                                                          // never addition-order rounding
    const bool onehot = getenv("Q53_ONEHOT") != nullptr; // q row = e_0 (one 1.0 at d=0),
                                                          // K[j][0] = 2^(j%4), rest zero: S(j) is
                                                          // exactly one K element -- m per split
                                                          // pins the Q/K addressing map itself
    auto gen = [exact]() {
        if (exact) {
            static const float pw[] = {1.f, 2.f, 4.f, 8.f, -1.f, -2.f, -4.f, -8.f};
            return (double)pw[rnd32() % 8];
        }
        double f; do f = rndu(-2.0, 2.0); while (f == 0.0f);
        return f;
    };
    if (onehot) {
        for (size_t i = 0; i < hq.size(); i++)
            hq[i] = ((i % (size_t)HD) == 0) ? f32_bf16(1.f) : f32_bf16(0.f);
        for (int b = 0; b < NB; b++)
            for (int v = 0; v < KVH; v++)
                for (int sl = 0; sl < BS; sl++) {
                    size_t base = (((size_t)b * KVH + v) * BS + sl) * SLOTW;
                    static const int p2[4] = {1, 2, 4, 8};
                    for (int d = 0; d < SLOTW; d++)
                        hkv[base + d] = (d == 0) ? f32_bf16((float)p2[(b * BS + sl) % 4])
                                                 : f32_bf16(0.f);   /* V all zeros: P.0 == 0 */
                }
        for (auto& v : hq) { (void)v; }
    } else {
        for (auto& v : hq)  v = f32_bf16((float)gen());
        for (auto& v : hkv) v = f32_bf16((float)gen());
    }
    std::vector<int32_t> hbt(NB), hsu(1, CTX), hcu(2);
    for (int i = 0; i < NB; i++) hbt[i] = i;
    hcu[0] = 0; hcu[1] = rows;

    uint16_t *dq, *dkv, *dout; int32_t *dbt, *dsu, *dcu; float* dscr;
    const size_t scr_bytes = (size_t)rows * QH * 64 * (HD + 2) * 4;
    die("mq", hipMalloc((void**)&dq, hq.size() * 2));
    die("mkv", hipMalloc((void**)&dkv, hkv.size() * 2));
    die("mo", hipMalloc((void**)&dout, hq.size() * 2));
    die("mbt", hipMalloc((void**)&dbt, NB * 4));
    die("msu", hipMalloc((void**)&dsu, 4));
    die("mcu", hipMalloc((void**)&dcu, 8));
    die("mscr", hipMalloc((void**)&dscr, scr_bytes));
    die("cq", hipMemcpy(dq, hq.data(), hq.size() * 2, hipMemcpyHostToDevice));
    die("ckv", hipMemcpy(dkv, hkv.data(), hkv.size() * 2, hipMemcpyHostToDevice));
    die("co", hipMemset(dout, 0, hq.size() * 2));
    die("cbt", hipMemcpy(dbt, hbt.data(), NB * 4, hipMemcpyHostToDevice));
    die("csu", hipMemcpy(dsu, hsu.data(), 4, hipMemcpyHostToDevice));
    die("ccu", hipMemcpy(dcu, hcu.data(), 8, hipMemcpyHostToDevice));
    die("cscr", hipMemset(dscr, 0x7f, scr_bytes));          // poison: every live partial must overwrite

    RadTensor t[8] = {};
    t[0].data = dq; t[0].dtype = RAD_BF16; t[0].rank = 3;
    t[0].shape[0] = rows; t[0].shape[1] = QH; t[0].shape[2] = HD;
    t[0].stride[0] = ROWW; t[0].stride[1] = HD; t[0].stride[2] = 1;
    t[1].data = dkv; t[1].dtype = RAD_BF16; t[1].rank = 4;
    t[1].shape[0] = NB; t[1].shape[1] = KVH; t[1].shape[2] = BS; t[1].shape[3] = SLOTW;
    t[1].stride[0] = (int64_t)KVH * BS * SLOTW; t[1].stride[1] = (int64_t)BS * SLOTW;
    t[1].stride[2] = SLOTW; t[1].stride[3] = 1;
    t[2].data = dbt; t[2].dtype = RAD_I32; t[2].rank = 2;
    t[2].shape[0] = 1; t[2].shape[1] = NB; t[2].stride[0] = NB; t[2].stride[1] = 1;
    t[3].data = dsu; t[3].dtype = RAD_I32; t[3].rank = 1; t[3].shape[0] = 1; t[3].stride[0] = 1;
    t[6].data = dcu; t[6].dtype = RAD_I32; t[6].rank = 1; t[6].shape[0] = 2; t[6].stride[0] = 1;
    t[7].data = dout; t[7].dtype = RAD_BF16; t[7].rank = 3;
    t[7].shape[0] = rows; t[7].shape[1] = QH; t[7].shape[2] = HD;
    t[7].stride[0] = ROWW; t[7].stride[1] = HD; t[7].stride[2] = 1;

    RadParam pv[] = {
        RAD_INT("q_len", rows), RAD_INT("head_dim", HD), RAD_INT("gqa", GQA),
        RAD_INT("block_size", BS), RAD_INT("causal", 1), RAD_INT("window", 0),
        RAD_INT("n_head", QH), RAD_INT("max_seqs", 1), RAD_INT("max_ctx", CTX),
    };
    RadArgs a{};
    a.t = t; a.n_t = 8; a.p = pv; a.n_p = sizeof pv / sizeof pv[0];
    a.scratch = dscr; a.scratch_bytes = scr_bytes;

    int rc = entry(&a, (RadStream)0);
    if (rc) { printf("ENTRY rc=%d\n", rc); return 3; }
    die("sync", hipDeviceSynchronize());

    std::vector<uint16_t> got((size_t)rows * ROWW);
    die("rb", hipMemcpy(got.data(), dout, got.size() * 2, hipMemcpyDeviceToHost));
    uint64_t yh = 1469598103934665603ULL;
    for (size_t i = 0; i < got.size(); i++) { yh ^= got[i]; yh *= 1099511628211ULL; }

    const int64_t pn = (int64_t)rows * QH;
    const size_t pf_n = (size_t)pn * SPL * (HD + 2);        // pso | psm | psl f32
    std::vector<float> pf(pf_n);
    die("rpf", hipMemcpy(pf.data(), dscr, pf_n * 4, hipMemcpyDeviceToHost));

    char path[512];
    snprintf(path, sizeof path, "/s2/%s.out", tag);   dump(path, got.data(), got.size() * 2);
    snprintf(path, sizeof path, "/s2/%s.pf",  tag);   dump(path, pf.data(), pf_n * 4);
    printf("DUMP tag=%s hd=%d rows=%d ctx=%d spl=%d yhash=%016llx\n",
           tag, HD, rows, CTX, SPL, (unsigned long long)yh);
    return 0;
}
