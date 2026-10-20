// attn_bench.hip -- prefill timing + cross-arm identity probe for the WMMA attention
// tile. The kbench fixture packs cannot reach the WMMA prefill geometry (the shipped
// pack is head_dim-256-only and the derived rows fixture derives only the decode
// band), so this tool drives the ROW ENTRY (r3d_rad_attn_paged through the .so, the
// same path the engine takes) over serve-shaped geometries: head_dim 128/64, gqa 8,
// bf16 KV, page-block table, causal.
//
// Arms, one process each, one binary: R3D_ATTN_V1=1 selects the v1 schedule (the
// default), R3D_ATTN_V2=1 the v2 256-thread schedule, R3D_ATTN_V3=1 the v3 thin
// 32-row block (env read once per process -- run once per arm). Same-binary/same-day
// pairing, the gemm DBUF promotion pattern. v3's arithmetic is v1's per-row chain, so
// v3 and v1 hashes MUST match -- hash equality is the gate, timing is the question.
// The tool prints one line per geometry:
// the min/median launch us, the effective KV-read GB/s (tiles_q * q_heads * ctx *
// slot bytes -- the metric COVERAGE.md's latency diagnosis is written in), and a
// FNV hash of the output. v1 and v2 fix the mma accumulation ORDER by construction,
// so equal hashes are the strongest identity evidence the geometry admits -- a
// scheduler change that changed a single rounding would show up here.
// Build/run (radiance-dev on the card):
//   hipcc -O1 -std=c++20 -I/opt/radiance/include/radiance -w tools/attn_bench.hip \
//     -o /tmp/ab -ldl
//   LIBR3D_SO=/work/buildN/radiance_home/kernels/libr3d.so /tmp/ab
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "rad_abi.h"

static uint64_t fnv(const std::vector<uint16_t>& v) {
    uint64_t h = 1469598103934665603ull;
    for (uint16_t x : v) { h ^= x; h *= 1099511628211ull; }
    return h;
}

int main() {
    const char* gl = getenv("AB_GEOMS") ? getenv("AB_GEOMS")
        : "128 64 1024,128 128 2048,128 981 8192,128 512 4096,64 512 4096";
    void* dl = dlopen(getenv("LIBR3D_SO") ? getenv("LIBR3D_SO")
                                          : "/work/build/radiance_home/kernels/libr3d.so", RTLD_NOW);
    if (!dl) { printf("dlopen: %s\n", dlerror()); return 1; }
    auto entry = (int (*)(const RadArgs*, RadStream))dlsym(dl, "r3d_rad_attn_paged");
    if (!entry) { printf("dlsym: %s\n", dlerror()); return 1; }

    const int QH = getenv("AB_QH") ? atoi(getenv("AB_QH")) : 16;
    const int KVH = getenv("AB_KVH") ? atoi(getenv("AB_KVH")) : 2;
    const int BS = 16;
    const int REPS = getenv("AB_REPS") ? atoi(getenv("AB_REPS")) : 15;

    char buf[512]; strncpy(buf, gl, sizeof buf - 1); buf[sizeof buf - 1] = 0;
    for (char* save = nullptr, *tok = strtok_r(buf, ",", &save); tok;
         tok = strtok_r(nullptr, ",", &save)) {
        int HD, QLEN, CTX;
        if (sscanf(tok, "%d %d %d", &HD, &QLEN, &CTX) != 3) { printf("bad geom %s\n", tok); continue; }
        const int GQA = QH / KVH, ROWW = QH * HD, SLOTW = 2 * HD;
        const int NB = (CTX + BS - 1) / BS;

        uint32_t st = 0x51ed2701u;
        auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };
        std::vector<uint16_t> hq((size_t)QLEN * ROWW), hkv((size_t)NB * KVH * BS * SLOTW);
        for (auto& v : hq)  v = (uint16_t)(rnd() & 0x3fff);   // small finite bf16 patterns
        for (auto& v : hkv) v = (uint16_t)(rnd() & 0x3fff);
        std::vector<int32_t> hbt(NB), hsu(1, CTX);
        for (int i = 0; i < NB; ++i) hbt[i] = i;

        uint16_t *dq, *dkv, *dout; int32_t *dbt, *dsu;
        hipMalloc(&dq, hq.size() * 2);   hipMemcpy(dq, hq.data(), hq.size() * 2, hipMemcpyHostToDevice);
        hipMalloc(&dkv, hkv.size() * 2); hipMemcpy(dkv, hkv.data(), hkv.size() * 2, hipMemcpyHostToDevice);
        hipMalloc(&dout, hq.size() * 2); hipMemset(dout, 0, hq.size() * 2);
        hipMalloc(&dbt, NB * 4);         hipMemcpy(dbt, hbt.data(), NB * 4, hipMemcpyHostToDevice);
        hipMalloc(&dsu, 4);              hipMemcpy(dsu, hsu.data(), 4, hipMemcpyHostToDevice);

        /* Absent optional operands: value-initialized RadTensors (data NULL). NEVER
         * uninitialized -- a garbage rank makes rad_tensor_numel read past the struct. */
        RadTensor t_q{}, t_kv{}, t_bt{}, t_su{}, t_kd{}, t_vd{}, t_cu{}, t_o{};
        t_q.data = dq; t_q.dtype = RAD_BF16; t_q.rank = 3;
        t_q.shape[0] = QLEN; t_q.shape[1] = QH; t_q.shape[2] = HD;
        t_q.stride[0] = ROWW; t_q.stride[1] = HD; t_q.stride[2] = 1;
        t_kv.data = dkv; t_kv.dtype = RAD_BF16; t_kv.rank = 4;
        t_kv.shape[0] = NB; t_kv.shape[1] = KVH; t_kv.shape[2] = BS; t_kv.shape[3] = SLOTW;
        t_kv.stride[0] = (int64_t)KVH * BS * SLOTW; t_kv.stride[1] = (int64_t)BS * SLOTW;
        t_kv.stride[2] = SLOTW; t_kv.stride[3] = 1;
        t_bt.data = dbt; t_bt.dtype = RAD_I32; t_bt.rank = 2;
        t_bt.shape[0] = 1; t_bt.shape[1] = NB; t_bt.stride[0] = NB; t_bt.stride[1] = 1;
        t_su.data = dsu; t_su.dtype = RAD_I32; t_su.rank = 1;
        t_su.shape[0] = 1; t_su.stride[0] = 1;
        t_o.data = dout; t_o.dtype = RAD_BF16; t_o.rank = 3;
        t_o.shape[0] = QLEN; t_o.shape[1] = QH; t_o.shape[2] = HD;
        t_o.stride[0] = ROWW; t_o.stride[1] = HD; t_o.stride[2] = 1;
        const RadTensor tensors[8] = { t_q, t_kv, t_bt, t_su, t_kd, t_vd, t_cu, t_o };
        RadParam pv[] = {
            RAD_INT("q_len", QLEN), RAD_INT("head_dim", HD), RAD_INT("gqa", GQA),
            RAD_INT("block_size", BS), RAD_INT("causal", 1), RAD_INT("window", 0),
            RAD_INT("n_head", QH), RAD_INT("max_seqs", 8), RAD_INT("max_ctx", CTX),
            RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", "bf16"),
        };
        RadArgs a{}; a.p = pv; a.n_p = sizeof pv / sizeof pv[0];
        a.t = tensors; a.n_t = 8;

        int rc = 0;
        for (int i = 0; i < 2 + REPS; ++i) rc |= entry(&a, (RadStream)0);
        hipDeviceSynchronize();
        if (rc) { printf("geom %s: entry rc=%d\n", tok, rc); continue; }

        hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
        float best = 1e30f, all = 0.0f;
        for (int i = 0; i < REPS; ++i) {
            hipEventRecord(e0);
            entry(&a, (RadStream)0);
            hipEventRecord(e1);
            hipEventSynchronize(e1);
            float ms = 0.0f; hipEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
            all += ms;
        }
        std::vector<uint16_t> got((size_t)QLEN * ROWW);
        hipMemcpy(got.data(), dout, got.size() * 2, hipMemcpyDeviceToHost);

        const double tiles_q = (QLEN + 63) / 64;
        const double kv_read = tiles_q * QH * (double)CTX * 2.0 * HD * 2.0;   // per launch
        const double gflops = 2.0 * 2.0 * tiles_q * (double)CTX * HD * QH     // QK+PV, causal ~half
                            * 0.5;
        printf("geom hd=%d qlen=%d ctx=%d: min_us=%.1f med_us=%.1f kv_gbs=%.1f gf/s=%.0f hash=%016llx\n",
               HD, QLEN, CTX, best * 1e3f, (all / REPS) * 1e3f,
               kv_read / (best * 1e-3) / 1e9,
               gflops / (best * 1e-3) / 1e9,
               (unsigned long long)fnv(got));
        fflush(stdout);
        hipFree(dq); hipFree(dkv); hipFree(dout); hipFree(dbt); hipFree(dsu);
    }
    return 0;
}
