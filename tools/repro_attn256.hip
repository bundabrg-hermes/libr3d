// repro_attn256.hip -- numeric gate for the head_dim-256 WMMA tile (issue #11 m2).
//
// No stock fixture reaches the 256 tile with RANDOM operands (the packs carry recorded
// real geometry; the random-operand law says all-zero or all-positive fixtures have
// twice hidden this repo's staging bugs), so this harness drives the row ENTRY
// (r3d_rad_attn_paged, the exact path the engine takes) over production hd256 geometry
// -- 16 q heads / 4 kv heads (gqa 4), block_size 16, bf16 cache -- with LCG-random
// signed bf16 operands and compares against ref_attn_paged's chain transcribed to the
// host (the second opinion, ported from refsrc/libref the same way repro_attn.hip
// ported the small case). Gates per case: rel_l2 < 5e-3 (kbench's tile gate is 2.5e-3;
// this adds the bf16-oracle rounding margin the attn_oracle_sweep law carries) AND no
// >0.1% outliers (bit-diff > 16 on a live element).
//
// Arms are process-scoped env (the .so reads levers once): run once with R3D_HD256
// unset (v1 scalar arm) and once with R3D_HD256=1 (tile arm) -- the per-case FNV
// hashes MUST differ between arms on every 256 case (a tile that secretly ran v1
// prints the same hash), MUST match between arms on the 128 control cases (the lever
// may not perturb the shipped geometry), and every run prints its DEVICE line: a run
// on the wrong card is phantom, period (see gate.sh's 1-device assertion).
// Cases fuzz: negative and per-head k_descale/v_descale (sign handling in the folded
// scale2 domain), unmapped (negative) block-table entries, sliding windows at 256,
// tile-edge straddling query lengths, ragged multi-sequence batches via cu_seqlens,
// and the decode/verify rows (1 and 4) the speculative serve actually runs.
//
// Build/run (radiance-dev, on the card; RAM-cheap: plain hipcc of one small TU):
//   hipcc -O1 -std=c++20 -I/opt/radiance/include/radiance -w tools/repro_attn256.hip \
//     -o /tmp/ra256 -ldl
//   LIBR3D_SO=/work/build_v117/radiance_home/kernels/libr3d.so /tmp/ra256            # v1 arm
//   R3D_HD256=1 LIBR3D_SO=/work/build_v117/radiance_home/kernels/libr3d.so /tmp/ra256 # tile arm
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

static uint64_t fnv(const std::vector<uint16_t>& v) {
    uint64_t h = 1469598103934665603ull;
    for (uint16_t x : v) { h ^= x; h *= 1099511628211ull; }
    return h;
}

static int fail_count = 0;

/* One case. n_seq sequences share a launch (cu_seqlens present when n_seq > 1 or the
 * caller forces one); ctx[] is the per-seq context; kdes/vdes when non-null carry
 * n_seq*KVH f32 scales (negatives included by the callers that pass them). */
static void run_case(int HD, int QLEN, const int* CTX_ARR, int NSEQ, int WINDOW,
                     int descale_mode,   // 0: absent; 1: positive varied; 2: signed incl. negative
                     int (*entry)(const RadArgs*, RadStream), const char* tag) {
    int DESCALE_MODE = descale_mode;
    const char* one = getenv("REPRO_ONE");
    if (one && strcmp(one, tag)) return;
    if (getenv("REPRO_DES")) DESCALE_MODE = atoi(getenv("REPRO_DES"));
    const int QH = 16, KVH = 4, BS = 16, GQA = QH / KVH;
    const int SLOTW = 2 * HD, ROWW = QH * HD;
    int ctxmax = 0, qlen_done = 0;
    int qlens[4], starts[4];
    for (int s = 0; s < NSEQ; ++s) {
        if (CTX_ARR[s] > ctxmax) ctxmax = CTX_ARR[s];
        starts[s] = qlen_done;
        // split QLEN across seqs, last seq takes the remainder (>= 1 each)
        qlens[s] = (s + 1 == NSEQ) ? QLEN - qlen_done
                                   : QLEN / NSEQ + (s == 0 && QLEN % NSEQ ? 1 : 0);
        if (qlens[s] < 1) qlens[s] = 1;
        qlen_done += qlens[s];
    }
    if (qlen_done != QLEN) { printf("SKIP %s: bad seq split\n", tag); ++fail_count; return; }
    if (QLEN > ctxmax) { printf("SKIP %s: qlen > ctx\n", tag); ++fail_count; return; }

    const int NB = (ctxmax + BS - 1) / BS;
    uint32_t st = 0x256ba5e5u + QLEN * 7919u + ctxmax * 104729u + (unsigned)WINDOW * 31u
                + (unsigned)DESCALE_MODE * 1000003u + (unsigned)HD;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };

    std::vector<uint16_t> hq((size_t)QLEN * ROWW), hkv((size_t)NB * KVH * BS * SLOTW);
    // signed, nonzero-random bf16 in [-2, 2): full sign bits, real mantissas (never
    // all-zero -- the trap twice burned this repo).
    for (auto& v : hq)  v = rtne_bf16(((rnd() & 0xffff) / 16384.0f) - 2.0f + 0.001f);
    for (auto& v : hkv) v = rtne_bf16(((rnd() & 0xffff) / 16384.0f) - 2.0f + 0.001f);
    std::vector<int32_t> hbt((size_t)NSEQ * NB);          // every seq row of the table
    for (size_t i = 0; i < hbt.size(); ++i) hbt[i] = (int32_t)(i % NB);
    if (NB >= 5) hbt[4] = -1;                             // unmapped mid-context, every
    if (NB >= 9) hbt[8] = -1;                             // seq row (rows share content)
    std::vector<int32_t> hsu(NSEQ);
    for (int s = 0; s < NSEQ; ++s) hsu[s] = CTX_ARR[s];
    std::vector<float> hkd, hvd;
    if (DESCALE_MODE) {
        hkd.resize((size_t)NSEQ * KVH); hvd.resize((size_t)NSEQ * KVH);
        for (size_t e = 0; e < hkd.size(); ++e) {
            float k = 0.5f + (rnd() & 1023) / 1024.0f, v = 0.5f + (rnd() & 1023) / 1024.0f;
            if (DESCALE_MODE == 2 && (e & 1)) k = -k;               // negative k_descale arm
            if (DESCALE_MODE == 2 && e && !(e & 2)) v = -v;         // negative v_descale arm
            hkd[e] = k; hvd[e] = v;
        }
    }

    // ---- host oracle: ref_attn_paged's chain, verbatim order (dot ascending d,
    // kds inside the dot factor, max over visible, exp double->float, acc ascending
    // key, one narrowing at the store). Independent of the tile. ----
    const float scale = 1.0f / sqrtf((float)HD);
    std::vector<uint16_t> want((size_t)QLEN * ROWW);
    {
        std::vector<int32_t> cusv(NSEQ + 1);
        for (int s = 0; s <= NSEQ; ++s) cusv[s] = s < NSEQ ? starts[s] : QLEN;
        int tok = 0;
        for (int s = 0; s < NSEQ; ++s) {
            const int ctx = CTX_ARR[s], qlen = qlens[s];
            for (int i = 0; i < qlen; ++i, ++tok) {
                const int pos = ctx - qlen + i;
                for (int h = 0; h < QH; ++h) {
                    const int kh = h / GQA;
                    const float kds = hkd.empty() ? 1.0f : hkd[(size_t)s * KVH + kh];
                    const float vds = hvd.empty() ? 1.0f : hvd[(size_t)s * KVH + kh];
                    float mx = -INFINITY;
                    // scores indexed by raw j in a per-row buffer (ctx a few thousand)
                    std::vector<float> sc(ctx, 0.0f);
                    for (int j = 0; j < ctx; ++j) {
                        if (j > pos) continue;
                        if (WINDOW > 0 && pos - j >= WINDOW) continue;
                        const int blk = hbt[(size_t)s * NB + j / BS];   // per-seq row view
                        if (blk < 0) continue;
                        const int64_t kb = (int64_t)blk * KVH * BS * SLOTW + (int64_t)kh * BS * SLOTW
                                         + (j % BS) * SLOTW;
                        float dot = 0;
                        for (int d = 0; d < HD; ++d)
                            dot += bf16_f(hq[(size_t)tok * ROWW + h * HD + d])
                                 * (bf16_f(hkv[kb + d]) * kds);
                        const float sco = dot * scale;
                        sc[j] = sco;                       // indexed by j: pass two reads
                        if (sco > mx) mx = sco;            // sc[jj] at the same j
                    }
                    float sum = 0, acc[256] = {};
                    size_t jj = 0;
                    for (int j = 0; j < ctx; ++j, ++jj) {
                        if (j > pos) continue;
                        if (WINDOW > 0 && pos - j >= WINDOW) continue;
                        const int blk = hbt[(size_t)s * NB + j / BS];   // per-seq row view
                        if (blk < 0) continue;
                        const float w = (float)std::exp((double)(sc[jj] - mx));
                        sum += w;
                        const int64_t kb = (int64_t)blk * KVH * BS * SLOTW + (int64_t)kh * BS * SLOTW
                                         + (j % BS) * SLOTW;
                        for (int d = 0; d < HD; ++d)
                            acc[d] += w * (bf16_f(hkv[kb + HD + d]) * vds);
                    }
                    (void)cusv;
                    const float inv = sum > 0 ? 1.0f / sum : 0.0f;
                    for (int d = 0; d < HD; ++d)
                        want[(size_t)tok * ROWW + h * HD + d] = rtne_bf16(acc[d] * inv);
                }
            }
        }
    }

    // ---- device buffers + ABI plumbing ----
    uint16_t *dq, *dkv, *dout; int32_t *dbt, *dsu, *dcu = nullptr; float *dkd = nullptr, *dvd = nullptr;
    hipMalloc(&dq, hq.size() * 2);   hipMemcpy(dq, hq.data(), hq.size() * 2, hipMemcpyHostToDevice);
    hipMalloc(&dkv, hkv.size() * 2); hipMemcpy(dkv, hkv.data(), hkv.size() * 2, hipMemcpyHostToDevice);
    hipMalloc(&dout, want.size() * 2); hipMemset(dout, 0, want.size() * 2);
    hipMalloc(&dbt, hbt.size() * 4);   hipMemcpy(dbt, hbt.data(), hbt.size() * 4, hipMemcpyHostToDevice);
    hipMalloc(&dsu, NSEQ * 4);       hipMemcpy(dsu, hsu.data(), NSEQ * 4, hipMemcpyHostToDevice);
    std::vector<int32_t> hcus;
    if (NSEQ > 1) {
        hcus.resize(NSEQ + 1);
        for (int s = 0; s < NSEQ; ++s) hcus[s] = starts[s];
        hcus[NSEQ] = QLEN;
        hipMalloc(&dcu, hcus.size() * 4); hipMemcpy(dcu, hcus.data(), hcus.size() * 4, hipMemcpyHostToDevice);
    }
    RadTensor t_kd{}, t_vd{};
    if (DESCALE_MODE) {
        hipMalloc(&dkd, hkd.size() * 4); hipMemcpy(dkd, hkd.data(), hkd.size() * 4, hipMemcpyHostToDevice);
        hipMalloc(&dvd, hvd.size() * 4); hipMemcpy(dvd, hvd.data(), hvd.size() * 4, hipMemcpyHostToDevice);
        t_kd.data = dkd; t_kd.dtype = RAD_F32; t_kd.rank = 2;
        t_kd.shape[0] = NSEQ; t_kd.shape[1] = KVH; t_kd.stride[0] = KVH; t_kd.stride[1] = 1;
        t_vd = t_kd; t_vd.data = dvd;
    }

    RadTensor t_q{}, t_kv{}, t_bt{}, t_su{}, t_cu{}, t_o{};
    t_q.data = dq; t_q.dtype = RAD_BF16; t_q.rank = 3;
    t_q.shape[0] = QLEN; t_q.shape[1] = QH; t_q.shape[2] = HD;
    t_q.stride[0] = ROWW; t_q.stride[1] = HD; t_q.stride[2] = 1;
    t_kv.data = dkv; t_kv.dtype = RAD_BF16; t_kv.rank = 4;
    t_kv.shape[0] = NB; t_kv.shape[1] = KVH; t_kv.shape[2] = BS; t_kv.shape[3] = SLOTW;
    t_kv.stride[0] = (int64_t)KVH * BS * SLOTW; t_kv.stride[1] = (int64_t)BS * SLOTW;
    t_kv.stride[2] = SLOTW; t_kv.stride[3] = 1;
    t_bt.data = dbt; t_bt.dtype = RAD_I32; t_bt.rank = 2;
    t_bt.shape[0] = NSEQ; t_bt.shape[1] = NB; t_bt.stride[0] = NB; t_bt.stride[1] = 1;
    t_su.data = dsu; t_su.dtype = RAD_I32; t_su.rank = 1;
    t_su.shape[0] = NSEQ; t_su.stride[0] = 1;
    if (dcu) { t_cu.data = dcu; t_cu.dtype = RAD_I32; t_cu.rank = 1;
               t_cu.shape[0] = NSEQ + 1; t_cu.stride[0] = 1; }
    t_o.data = dout; t_o.dtype = RAD_BF16; t_o.rank = 3;
    t_o.shape[0] = QLEN; t_o.shape[1] = QH; t_o.shape[2] = HD;
    t_o.stride[0] = ROWW; t_o.stride[1] = HD; t_o.stride[2] = 1;
    // The v1 kernel dereferences the k_descale/v_descale RadTensor structs ON DEVICE
    // (ldt_dev reads t->dtype/t->data in-kernel) and the engine lives in managed
    // memory. A host-stack descriptor array is not device-visible: the kernel faults
    // on it (bisected 2026-10-24: every descale case died, descale-0 passed). Place
    // the whole tensor array in host-mapped memory, as the engine's own arrays are.
    RadTensor* tensors = nullptr;
    hipHostMalloc(&tensors, 8 * sizeof(RadTensor), hipHostMallocMapped);
    tensors[0] = t_q; tensors[1] = t_kv; tensors[2] = t_bt; tensors[3] = t_su;
    tensors[4] = t_kd; tensors[5] = t_vd; tensors[6] = t_cu; tensors[7] = t_o;
    RadParam pv[] = {
        RAD_INT("q_len", QLEN), RAD_INT("head_dim", HD), RAD_INT("gqa", GQA),
        RAD_INT("block_size", BS), RAD_INT("causal", 1), RAD_INT("window", WINDOW),
        RAD_INT("n_head", QH), RAD_INT("max_seqs", 8), RAD_INT("max_ctx", ctxmax),
        RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", "bf16"),
    };
    RadArgs a{}; a.p = pv; a.n_p = sizeof pv / sizeof pv[0];
    a.t = tensors; a.n_t = 8;

    int rc = entry(&a, (RadStream)0);
    hipDeviceSynchronize();
    std::vector<uint16_t> got(want.size());
    hipMemcpy(got.data(), dout, want.size() * 2, hipMemcpyDeviceToHost);

    double num = 0, den = 0; int worst = 0, outliers = 0;
    for (size_t e = 0; e < want.size(); ++e) {
        const float wv = bf16_f(want[e]), gv = bf16_f(got[e]);
        num += (double)(gv - wv) * (gv - wv);
        den += (double)wv * wv;
        const int du = std::abs((int)want[e] - (int)got[e]);
        if (du > worst && std::isfinite(wv) && std::isfinite(gv)) worst = du;
        if (du > 16 && std::fabs(wv) > 1e-2f) ++outliers;
    }
    const double rel = den > 0 ? sqrt(num / den) : (num > 0 ? 1.0 : 0.0);
    const bool ok = rc == 0 && rel < 5e-3 && outliers * 1000 <= (int)want.size();
    if (getenv("REPRO_DBG")) {                       // row-localization for bisect
        for (int t = 0; t < QLEN; ++t) {
            double rn = 0, rd = 0;
            for (int h = 0; h < QH; ++h)
                for (int d = 0; d < HD; ++d) {
                    size_t e = (size_t)t * ROWW + h * HD + d;
                    rn += pow(bf16_f(got[e]) - bf16_f(want[e]), 2); rd += pow(bf16_f(want[e]), 2);
                }
            if (rd > 0 && sqrt(rn / rd) > (getenv("REPRO_DBG2") ? 0.0 : 1e-2)) {
                double best = 1e30; int bt2 = -1;
                if (getenv("REPRO_DBG2")) {   // best-matching want row (shift detector)
                    for (int u = 0; u < QLEN; ++u) {
                        double sn = 0, sd = 0;
                        for (int h = 0; h < QH; ++h)
                            for (int d = 0; d < HD; ++d) {
                                size_t e1 = (size_t)t * ROWW + h * HD + d;
                                size_t e2 = (size_t)u * ROWW + h * HD + d;
                                sn += pow(bf16_f(got[e1]) - bf16_f(want[e2]), 2);
                                sd += pow(bf16_f(want[e2]), 2);
                            }
                        double sr = sd > 0 ? sqrt(sn / sd) : sn;
                        if (sr < best) { best = sr; bt2 = u; }
                    }
                }
                printf("  DBG row %d rel=%.3e bestmatch=%d(%.1e)\n", t, sqrt(rn / rd), bt2, best);
            }
        }
    }
    printf("%s %s hd=%d qlen=%d ctx=%d(+%d) nseq=%d win=%d descale=%d: rc=%d rel_l2=%.3e worstULP=%d outliers=%d hash=%016llx\n",
           ok ? "PASS" : "FAIL", tag, HD, QLEN, CTX_ARR[0], NSEQ > 1 ? CTX_ARR[1] : 0,
           NSEQ, WINDOW, DESCALE_MODE, rc, rel, worst, outliers,
           (unsigned long long)fnv(got));
    fflush(stdout);
    if (!ok) ++fail_count;
    hipFree(dq); hipFree(dkv); hipFree(dout); hipFree(dbt); hipFree(dsu);
    hipHostFree(tensors);
    if (dcu) hipFree(dcu);
    if (dkd) hipFree(dkd);
    if (dvd) hipFree(dvd);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    void* dl = dlopen(getenv("LIBR3D_SO") ? getenv("LIBR3D_SO")
                                          : "/work/build_v117/radiance_home/kernels/libr3d.so", RTLD_NOW);
    if (!dl) { printf("dlopen: %s\n", dlerror()); return 1; }
    auto entry = (int (*)(const RadArgs*, RadStream))dlsym(dl, "r3d_rad_attn_paged");
    if (!entry) { printf("dlsym: %s\n", dlerror()); return 1; }

    // The device assertion: this run is only real on the XTX (gfx1100).
    int dev = -1; hipDeviceProp_t prop{};
    if (hipGetDevice(&dev) == hipSuccess && hipGetDeviceProperties(&prop, dev) == hipSuccess)
        printf("device: %s (index %d, gcnArchName=%s, vram=%lld MiB)\n",
               prop.name, dev, prop.gcnArchName, (long long)(prop.totalGlobalMem >> 20));
    else { printf("device: QUERY FAILED -- phantom run, aborting\n"); return 1; }

    const int c1[]  = { 512 };
    const int c2[]  = { 2048 };
    const int c3[]  = { 130 };
    const int c4[]  = { 260 };
    const int c5[]  = { 600, 350 };
    const int c6[]  = { 8192 };
    const int c7[]  = { 1024 };

    // 256-class cases (tile iff R3D_HD256; v1 otherwise -- hashes split the arms):
    run_case(256, 1,   c1, 1, 0, 1, entry, "decode-r1");
    run_case(256, 4,   c2, 1, 0, 2, entry, "verify-r4-neg");   // spec-3 band, signed descales
    run_case(256, 70,  c3, 1, 0, 0, entry, "tile-edge");       // straddles the 64-row tile
    run_case(256, 130, c4, 1, 64, 1, entry, "window-win");     // sliding window at 256
    run_case(256, 66,  c5, 2, 0, 2, entry, "ragged-2seq-neg"); // cu_seqlens batch, signed
    run_case(256, 1,   c6, 1, 0, 1, entry, "decode-r1-deep");  // 8k context
    run_case(256, 5,   c7, 1, 0, 2, entry, "verify-r5-neg");   // block-pass band (<= 8)
    // 128 control: must hash EQUAL between lever arms (the lever may not perturb 128).
    run_case(128, 70,  c3, 1, 0, 1, entry, "ctl128-edge");
    run_case(128, 4,   c1, 1, 0, 2, entry, "ctl128-neg");

    printf(fail_count ? "REPRO256 FAIL: %d case(s)\n" : "REPRO256 PASS\n", fail_count);
    return fail_count != 0;
}
