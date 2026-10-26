// bench_attn_decode.hip -- decode-band bench for libr3d's r3d_rad_attn_paged_decode
// row (split-KV WMMA slice + merge, GQA fold, DBUF staging), driven through dlopen
// of the same libr3d.so the serve loads. Forked from the Phase C bench_r3d.hip with
// the single generalization issue #11 milestone 3 needs: head_dim is an env lever
// (BENCH_HD, default 128 -> every v148-era command reproduces verbatim; BENCH_HD=256
// + R3D_HD256=1 drives the new slice arm). Shapes: bf16 KV, page block 16, arbitrary
// QH/KVH GQA, rows 1..8 (decode + speculative verify band), n_seq 1 + cu_seqlens as
// serve passes them. GB/s accounting: KV bytes streamed once per kv-head group by
// the fold arm = ctx * kvh * 2(K+V) * hd * 2 B (the plain arm streams gqa times that
// -- raw med_us is the arm-neutral number; the DBUF gate needs contiguous 16B KV,
// which this harness's identity page table always gives).
// usage: bench_attn_decode dev
//        bench_attn_decode parity <qh> <kvh> <rows> <ctx>
//        bench_attn_decode time   <qh> <kvh> <rows> <ctx> <reps> <sets>
#include <hip/hip_runtime.h>
#include <dlfcn.h>

#include <algorithm>
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
static uint32_t xs = 0x2545f491u;
static uint32_t rnd32() { xs ^= xs << 13; xs ^= xs >> 17; xs ^= xs << 5; return xs; }
static double rndu(double lo, double hi) { return lo + (hi - lo) * (rnd32() * (1.0 / 4294967296.0)); }

static float bf16_val(uint16_t h) {
    union { uint32_t u; float f; } cv; cv.u = (uint32_t)h << 16; return cv.f;
}
static uint16_t f32_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

static void assert_xtx() {
    hipDeviceProp_t p;
    die("props", hipGetDeviceProperties(&p, 0));
    printf("DEV name=%s arch=%s memGB=%.1f\n", p.name, p.gcnArchName,
           p.totalGlobalMem / 1073741824.0);
    fflush(stdout);
    if (!strstr(p.gcnArchName, "gfx1100")) { printf("ABORT not-gfx1100\n"); exit(3); }
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "dev";
    if (!strcmp(mode, "dev")) {
        int n = 0; die("getcount", hipGetDeviceCount(&n));
        for (int i = 0; i < n; i++) {
            hipDeviceProp_t p; die("props", hipGetDeviceProperties(&p, i));
            printf("DEV idx=%d name=%s arch=%s\n", i, p.name, p.gcnArchName);
        }
        return 0;
    }
    assert_xtx();
    void* dl = dlopen(getenv("LIBR3D_SO") ? getenv("LIBR3D_SO")
                                          : "/rh/kernels/libr3d.so", RTLD_NOW);
    if (!dl) { printf("dlopen: %s\n", dlerror()); return 1; }
    auto entry = (int (*)(const RadArgs*, RadStream))dlsym(dl, "r3d_rad_attn_paged_decode");
    if (!entry) { printf("dlsym: %s\n", dlerror()); return 1; }

    const int QH = atoi(argv[2]), KVH = atoi(argv[3]);
    const int rows = atoi(argv[4]), CTX = atoi(argv[5]);
    const int reps = argc > 6 ? atoi(argv[6]) : 11;
    const int sets = argc > 7 ? atoi(argv[7]) : 8;
    const int HD = getenv("BENCH_HD") ? atoi(getenv("BENCH_HD")) : 128;
    const int GQA = QH / KVH, BS = 16;
    const int NB = (CTX + BS - 1) / BS;
    const int ROWW = QH * HD, SLOTW = 2 * HD;

    /* random non-zero bf16 operands */
    std::vector<uint16_t> hq((size_t)rows * ROWW), hkv((size_t)NB * KVH * BS * SLOTW);
    for (auto& v : hq)  { float f; do f = (float)rndu(-2.0, 2.0); while (f == 0.0f); v = f32_bf16(f); }
    for (auto& v : hkv) { float f; do f = (float)rndu(-2.0, 2.0); while (f == 0.0f); v = f32_bf16(f); }
    std::vector<int32_t> hbt(NB), hsu(1, CTX), hcu(2);
    for (int i = 0; i < NB; i++) hbt[i] = i;   /* identity page table, all mapped */
    hcu[0] = 0; hcu[1] = rows;

    uint16_t *dq, *dkv, *dout; int32_t *dbt, *dsu, *dcu; float* dscr;
    const size_t scr_bytes = (size_t)rows * QH * 64 * (HD + 2) * 4;   /* splits law caps at 64 */
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
    /* t[4] kdes, t[5] vdes: absent (data NULL, value-initialized) */
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
    if (rc) { printf("ENTRY rc=%d (unsupported/device)\n", rc); return 3; }
    die("sync", hipDeviceSynchronize());

    if (!strcmp(mode, "parity")) {
        std::vector<uint16_t> got((size_t)rows * ROWW);
        die("rb", hipMemcpy(got.data(), dout, got.size() * 2, hipMemcpyDeviceToHost));
        double sd = 0, sr = 0, smax = 0; int bad = 0;
        for (int r = 0; r < rows; r++)
            for (int h = 0; h < QH; h++) {
                const int kvh = h / GQA, pos = CTX - rows + r;   /* causal pos, as the row */
                std::vector<double> o(HD, 0.0);
                double mx = -1e300;
                std::vector<double> s(pos + 1);
                for (int j = 0; j <= pos; j++) {
                    const int blk = hbt[j / BS];
                    const uint16_t* krow = &hkv[(((size_t)blk * KVH + kvh) * BS + (j % BS)) * SLOTW];
                    const uint16_t* qrow = &hq[(size_t)r * ROWW + (size_t)h * HD];
                    double acc = 0;
                    for (int d = 0; d < HD; d++) acc += (double)bf16_val(qrow[d]) * bf16_val(krow[d]);
                    s[j] = acc / sqrt((double)HD);
                    if (s[j] > mx) mx = s[j];
                }
                double den = 0;
                for (int j = 0; j <= pos; j++) {
                    double p = exp(s[j] - mx); den += p;
                    const int blk = hbt[j / BS];
                    const uint16_t* vrow = &hkv[(((size_t)blk * KVH + kvh) * BS + (j % BS)) * SLOTW + HD];
                    for (int d = 0; d < HD; d++) o[d] += p * bf16_val(vrow[d]);
                }
                for (int d = 0; d < HD; d++) {
                    double ref = o[d] / den, g = bf16_val(got[(size_t)r * ROWW + (size_t)h * HD + d]);
                    if (!std::isfinite(g)) bad++;
                    double e = g - ref; sd += e * e; sr += ref * ref;
                    if (fabs(e) > smax) smax = fabs(e);
                }
            }
        double rel = sqrt(sd / sr);
        printf("PARITY arm=r3d hd=%d qh=%d kvh=%d rows=%d ctx=%d rel_l2=%.6f rel_max=%.6g nonfinite=%d %s\n",
               HD, QH, KVH, rows, CTX, rel, smax, bad, (rel < 0.01 && !bad) ? "PASS" : "FAIL");
        return (rel < 0.01 && !bad) ? 0 : 4;
    }

    /* timed arm: rotate kv sets (L2 defeat). t[1].data swap per launch. */
    std::vector<uint16_t*> dkvsets(sets, nullptr);
    dkvsets[0] = dkv;
    for (int s = 1; s < sets; s++) {
        die("mkv2", hipMalloc((void**)&dkvsets[s], hkv.size() * 2));
        die("ckv2", hipMemcpy(dkvsets[s], dkv, hkv.size() * 2, hipMemcpyHostToDevice));
    }
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    std::vector<float> ms;
    for (int i = 0; i < reps; i++) {
        t[1].data = dkvsets[i % sets];
        hipEventRecord(e0);
        rc = entry(&a, (RadStream)0);
        hipEventRecord(e1);
        if (rc) { printf("launch %d rc=%d\n", i, rc); return 3; }
        die("esync", hipEventSynchronize(e1));
        float tt; hipEventElapsedTime(&tt, e0, e1);
        ms.push_back(tt);
    }
    std::sort(ms.begin(), ms.end());
    double med = ms[ms.size() / 2], mn = ms.front(), mx = ms.back();
    double kvb = (double)CTX * KVH * 2 * HD * 2.0;   /* fold arm: once per group */
    printf("RESULT arm=r3d hd=%d qh=%d kvh=%d rows=%d ctx=%d sets=%d reps=%d med_us=%.1f min_us=%.1f "
           "max_us=%.1f kv_MB=%.1f gbs_kv=%.1f\n",
           HD, QH, KVH, rows, CTX, sets, reps, med * 1e3, mn * 1e3, mx * 1e3,
           kvb / 1e6, kvb / (med * 1e-3) / 1e9);
    return 0;
}
