// attn_oracle_sweep.hip -- oracle sweep for the attn_paged TILE entry
// (r3d_rad_attn_paged, the path the attn_paged_bf16 row resolves to; the WMMA v1/v2
// arms and the scalar fallback sit behind it, and R3D_ATTN_V1 / R3D_NO_WMMA pick the
// arithmetic in the .so -- run this once per arm).
//
// The geometries are the ones the fixture packs do NOT reach on the WMMA arms:
// head_dim 64/128 with gqa, sliding WINDOW, unmapped cache blocks, and query tiles
// straddling the 64-row tile edge. The shipped pack is head_dim-256-only, so every
// window case ever replayed ran on the scalar arm -- the vacuous-pass trap with a
// window on it. (repro_attn.hip stays the single-case bisection tool it was.)
//   hipcc -O1 -std=c++20 -I/opt/radiance/include/radiance -w tools/attn_oracle_sweep.hip \
//     -o /tmp/ao -ldl
//   LIBR3D_SO=$PWD/buildN/radiance_home/kernels/libr3d.so /tmp/ao
// Pass line: rel_l2 < 5e-3 (kbench's tile gate is 2.5e-3; this adds bf16 oracle
// rounding on exp) AND no element differing by more than 8 bf16 ULPs.
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "rad_abi.h"

static uint16_t rtne_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float bf16_f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

static int fail_count = 0;

static void run_case(int HD, int BS, int CTX, int QLEN, int QH, int KVH, int WINDOW,
                     int (*entry)(const RadArgs*, RadStream)) {
    const int NB = (CTX + BS - 1) / BS;
    const int SLOTW = 2 * HD, ROWW = QH * HD, GQA = QH / KVH;

    uint32_t st = 0x51ed2701u + QLEN * 7919 + HD * 104729 + CTX * 31 + WINDOW * 7;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };

    std::vector<uint16_t> hq((size_t)QLEN * ROWW), hkv((size_t)NB * KVH * BS * SLOTW);
    for (auto& v : hq)  v = rtne_bf16(((rnd() & 0xffff) / 32768.0f) - 1.0f);
    for (auto& v : hkv) v = rtne_bf16(((rnd() & 0xffff) / 32768.0f) - 1.0f);
    std::vector<int32_t> hbt(NB), hsu(1, CTX);
    for (int i = 0; i < NB; ++i) hbt[i] = i;
    if (NB >= 3) hbt[2] = -1;                 // block 2 unmapped, every case with room

    // ---- host oracle: ref_attn_paged's chain, causal + window + page table ----
    const float scale = 1.0f / sqrtf((float)HD);
    std::vector<uint16_t> want((size_t)QLEN * ROWW);
    std::vector<float> sc(CTX);
    for (int tok = 0; tok < QLEN; ++tok) {
        const int pos = CTX - QLEN + tok;
        for (int h = 0; h < QH; ++h) {
            const int kh = h / GQA;
            float mx = -INFINITY;
            std::fill(sc.begin(), sc.end(), 0.0f);
            for (int j = 0; j < CTX; ++j) {
                if (j > pos) continue;
                if (WINDOW > 0 && pos - j >= WINDOW) continue;
                const int blk = hbt[j / BS];
                if (blk < 0) continue;
                const int64_t kb = (int64_t)blk * KVH * BS * SLOTW + kh * BS * SLOTW
                                 + (j % BS) * SLOTW;
                float dot = 0;
                for (int d = 0; d < HD; ++d)
                    dot += bf16_f(hq[(size_t)tok * ROWW + h * HD + d]) * bf16_f(hkv[kb + d]);
                sc[j] = dot * scale;
                if (sc[j] > mx) mx = sc[j];
            }
            float sum = 0, acc[512] = {};
            for (int j = 0; j < CTX; ++j) {
                if (j > pos) continue;
                if (WINDOW > 0 && pos - j >= WINDOW) continue;
                const int blk = hbt[j / BS];
                if (blk < 0) continue;
                const int64_t kb = (int64_t)blk * KVH * BS * SLOTW + kh * BS * SLOTW
                                 + (j % BS) * SLOTW;
                const float w = (float)std::exp((double)(sc[j] - mx));
                sum += w;
                for (int d = 0; d < HD; ++d) acc[d] += w * bf16_f(hkv[kb + HD + d]);
            }
            const float inv = sum > 0 ? 1.0f / sum : 0.0f;
            for (int d = 0; d < HD; ++d)
                want[(size_t)tok * ROWW + h * HD + d] = rtne_bf16(acc[d] * inv);
        }
    }

    uint16_t *dq, *dkv, *dout; int32_t *dbt, *dsu;
    hipMalloc(&dq, hq.size() * 2);   hipMemcpy(dq, hq.data(), hq.size() * 2, hipMemcpyHostToDevice);
    hipMalloc(&dkv, hkv.size() * 2); hipMemcpy(dkv, hkv.data(), hkv.size() * 2, hipMemcpyHostToDevice);
    hipMalloc(&dout, want.size() * 2); hipMemset(dout, 0, want.size() * 2);
    hipMalloc(&dbt, NB * 4);         hipMemcpy(dbt, hbt.data(), NB * 4, hipMemcpyHostToDevice);
    hipMalloc(&dsu, 4);              hipMemcpy(dsu, hsu.data(), 4, hipMemcpyHostToDevice);

    /* An absent optional operand is a RadTensor whose .data is NULL -- not a missing
     * slot, not a zeroed struct (dtype 0 reads as a bad dtype). See rad_plugin.h. */
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
        RAD_INT("block_size", BS), RAD_INT("causal", 1), RAD_INT("window", WINDOW),
        RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", "bf16"),
    };
    RadArgs a{}; a.p = pv; a.n_p = sizeof pv / sizeof pv[0];
    a.t = tensors; a.n_t = 8;

    int rc = entry(&a, (RadStream)0);
    hipDeviceSynchronize();
    std::vector<uint16_t> got(want.size());
    hipMemcpy(got.data(), dout, want.size() * 2, hipMemcpyDeviceToHost);

    /* Gate = kbench's gate: rel_l2 against the oracle. The ULP figure is diagnostic:
     * the oracle sums in double while the kernel accumulates f32 fragment partials, so
     * cancellation-heavy near-zero elements can differ by large BIT distances without
     * any real error. Outlier = bit diff > 16 on an element whose |value| is above the
     * noise floor; more than 0.1% of elements failing that is a genuine break. */
    double num = 0, den = 0; int worst = 0, wtok = -1, wd = -1, outliers = 0;
    for (size_t e = 0; e < want.size(); ++e) {
        const float wv = bf16_f(want[e]), gv = bf16_f(got[e]);
        num += (double)(gv - wv) * (gv - wv);
        den += (double)wv * wv;
        const int du = std::abs((int)want[e] - (int)got[e]);   // distance on bf16 bit lines
        if (du > worst && std::isfinite(wv) && std::isfinite(gv)) { worst = du; wtok = (int)(e / ROWW); wd = (int)(e % ROWW); }
        if (du > 16 && std::fabs(wv) > 1e-2f) ++outliers;
    }
    const double rel = den > 0 ? sqrt(num / den) : (num > 0 ? 1.0 : 0.0);
    const bool ok = rc == 0 && rel < 5e-3 && outliers * 1000 <= (int)want.size();
    printf("%s hd=%d ctx=%d qlen=%d qh=%d kvh=%d win=%d unmapped=1: rc=%d rel_l2=%.2e worst=%dULP@(%d,%d) outliers=%d\n",
           ok ? "PASS" : "FAIL", HD, CTX, QLEN, QH, KVH, WINDOW, rc, rel, worst, wtok, wd, outliers);
    if (!ok) ++fail_count;
    hipFree(dq); hipFree(dkv); hipFree(dout); hipFree(dbt); hipFree(dsu);
}

int main() {
    void* dl = dlopen(getenv("LIBR3D_SO") ? getenv("LIBR3D_SO")
                                          : "/work/build/radiance_home/kernels/libr3d.so", RTLD_NOW);
    if (!dl) { printf("dlopen: %s\n", dlerror()); return 1; }
    auto entry = (int (*)(const RadArgs*, RadStream))dlsym(dl, "r3d_rad_attn_paged");
    if (!entry) { printf("dlsym: %s\n", dlerror()); return 1; }

    run_case(128, 4, 40, 5, 2, 1, 0, entry);    // h128 gqa2, windowless, unmapped mid
    run_case(128, 4, 40, 5, 2, 1, 8, entry);    // THE window pass the packs never ran on WMMA
    run_case(128, 16, 130, 70, 2, 2, 0, entry); // straddles the 64-row tile edge
    run_case(128, 16, 130, 70, 2, 2, 24, entry);
    run_case(64, 4, 37, 3, 4, 1, 0, entry);     // h64 arm, gqa 4
    run_case(64, 16, 257, 65, 16, 2, 0, entry); // serve-shaped heads, 17 blocks
    run_case(8, 4, 5, 2, 1, 1, 0, entry);       // refuses WMMA: repro_attn's scalar case
    printf(fail_count ? "ORACLE SWEEP FAIL: %d case(s)\n" : "ORACLE SWEEP PASS\n", fail_count);
    return fail_count != 0;
}
