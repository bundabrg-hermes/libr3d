// repro_attn_splitk.hip -- split-KV decode probe: runs the decode ROW entry on a small
// hand-built case, then separates the two halves of the mechanism on the host:
//   (a) the kernel's own out vs the oracle      -> the device arm end to end
//   (b) a host re-merge of the arena partials   -> are the partials what the contract says
// If (a) fails and (b) passes, the merge kernel is wrong. If (b) fails, the slice blocks
// (or their partial store) are. Prints the per-(head,split) m/l table so empty/inert
// slices and their (m,l) payloads are visible, and flags NaN/garbage slots.
//
// Build/run (in radiance-dev on the card):
//   hipcc -O1 -std=c++20 -I/opt/radiance/include/radiance -w tools/repro_attn_splitk.hip \
//     -o /tmp/rsk -ldl
//   LIBR3D_SO=$PWD/build6/radiance_home/kernels/libr3d.so R3D_FORCE_SPLITS=2 /tmp/rsk
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
static int envid(const char* k, int dflt) { const char* v = getenv(k); return v ? atoi(v) : dflt; }

int main() {
    const int HD = envid("REPRO_HD", 128), BS = 16, QH = 8, KVH = 2, GQA = 4;
    const int QLEN = envid("REPRO_QLEN", 1);
    const int CTX  = envid("REPRO_CTX", 300);
    const int NB   = (CTX + BS - 1) / BS;
    const int ROWW = QH * HD;
    const int SPL  = envid("REPRO_SPL", 2);            // mirrored into R3D_FORCE_SPLITS
    setenv("R3D_FORCE_SPLITS", getenv("REPRO_SPL") ? getenv("REPRO_SPL") : "2", 1);

    uint32_t st = 0x1357bd9fu;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };

    std::vector<uint16_t> hq((size_t)QLEN * ROWW);
    for (auto& v : hq) v = rtne_bf16(((rnd() & 0xffff) / 32768.0f) - 1.0f);
    std::vector<uint16_t> hkv((size_t)NB * KVH * BS * 2 * HD);   // K then V, bf16
    for (auto& v : hkv) v = rtne_bf16(((rnd() & 0xffff) / 32768.0f) - 1.0f);
    std::vector<int32_t> hbt(NB); for (int i = 0; i < NB; ++i) hbt[i] = i;
    int32_t hsu[1] = { CTX };

    const float scale = (float)(1.0 / sqrt((double)HD));

    // ---- oracle (plain f32 softmax, exact keys, all mapped) ----
    std::vector<uint16_t> want((size_t)QLEN * ROWW);
    for (int tok = 0; tok < QLEN; ++tok) {
        const int64_t pos = CTX - QLEN + tok;
        for (int h = 0; h < QH; ++h) {
            const int kh = h / GQA;
            std::vector<float> sc;
            float mx = -INFINITY;
            for (int64_t j = 0; j <= pos; ++j) {
                const int blk = hbt[j / BS];
                const int64_t kb = ((int64_t)blk * KVH + kh) * BS * 2 * HD + (j % BS) * 2 * HD;
                float dot = 0;
                for (int dd = 0; dd < HD; ++dd)
                    dot += bf16_f(hq[(size_t)tok * ROWW + h * HD + dd]) * bf16_f(hkv[kb + dd]);
                sc.push_back(dot * scale);
                if (sc.back() > mx) mx = sc.back();
            }
            float sum = 0; std::vector<float> w(sc.size());
            for (size_t i = 0; i < sc.size(); ++i) { w[i] = expf(sc[i] - mx); sum += w[i]; }
            float acc[256] = {};
            int jj = 0;
            for (int64_t j = 0; j <= pos; ++j) {
                const int blk = hbt[j / BS];
                const int64_t kb = ((int64_t)blk * KVH + kh) * BS * 2 * HD + (j % BS) * 2 * HD + HD;
                for (int dd = 0; dd < HD; ++dd) acc[dd] += w[jj] * bf16_f(hkv[kb + dd]);
                ++jj;
            }
            for (int dd = 0; dd < HD; ++dd)
                want[(size_t)tok * ROWW + h * HD + dd] = rtne_bf16(acc[dd] / sum);
        }
    }

    // ---- device + plugin ----
    void* dl = dlopen(getenv("LIBR3D_SO") ? getenv("LIBR3D_SO")
                                          : "/work/build6/radiance_home/kernels/libr3d.so", RTLD_NOW);
    if (!dl) { printf("dlopen: %s\n", dlerror()); return 1; }
    auto entry = (int (*)(const RadArgs*, RadStream))dlsym(dl, "r3d_rad_attn_paged_decode");
    if (!entry) { printf("dlsym: %s\n", dlerror()); return 1; }

    uint16_t *dq, *dout; uint16_t *dkv; int32_t *dbt, *dsu;
    hipMalloc(&dq, hq.size() * 2);   hipMemcpy(dq, hq.data(), hq.size() * 2, hipMemcpyHostToDevice);
    hipMalloc(&dkv, hkv.size() * 2); hipMemcpy(dkv, hkv.data(), hkv.size() * 2, hipMemcpyHostToDevice);
    hipMalloc(&dout, want.size() * 2); hipMemset(dout, 0, want.size() * 2);
    hipMalloc(&dbt, NB * 4);         hipMemcpy(dbt, hbt.data(), NB * 4, hipMemcpyHostToDevice);
    hipMalloc(&dsu, 4);              hipMemcpy(dsu, hsu, 4, hipMemcpyHostToDevice);

    // arena: the contract's worst case for this shape, pre-poisoned with NaN so any slot
    // the kernel failed to store reads as garbage instead of hiding behind zeros.
    const int64_t need = (int64_t)QLEN * QH * SPL * (HD + 2) * 4;
    void* dscratch = nullptr;
    hipMalloc(&dscratch, need);
    std::vector<uint8_t> poison(need, 0xFF);
    hipMemcpy(dscratch, poison.data(), need, hipMemcpyHostToDevice);

    RadTensor t_q{}, t_kv{}, t_bt{}, t_su{}, nul{}, cu{}, t_o{};
    t_q.data = dq; t_q.dtype = RAD_BF16; t_q.rank = 3;
    t_q.shape[0] = QLEN; t_q.shape[1] = QH; t_q.shape[2] = HD;
    t_q.stride[0] = ROWW; t_q.stride[1] = HD; t_q.stride[2] = 1;
    t_kv.data = dkv; t_kv.dtype = RAD_BF16; t_kv.rank = 4;
    t_kv.shape[0] = NB; t_kv.shape[1] = KVH; t_kv.shape[2] = BS; t_kv.shape[3] = 2 * HD;
    t_kv.stride[0] = (int64_t)KVH * BS * 2 * HD; t_kv.stride[1] = (int64_t)BS * 2 * HD;
    t_kv.stride[2] = 2 * HD; t_kv.stride[3] = 1;
    t_bt.data = dbt; t_bt.dtype = RAD_I32; t_bt.rank = 2;
    t_bt.shape[0] = 1; t_bt.shape[1] = NB; t_bt.stride[0] = NB; t_bt.stride[1] = 1;
    t_su.data = dsu; t_su.dtype = RAD_I32; t_su.rank = 1; t_su.shape[0] = 1; t_su.stride[0] = 1;
    t_o.data = dout; t_o.dtype = RAD_BF16; t_o.rank = 3;
    t_o.shape[0] = QLEN; t_o.shape[1] = QH; t_o.shape[2] = HD;
    t_o.stride[0] = ROWW; t_o.stride[1] = HD; t_o.stride[2] = 1;
    RadTensor host_arr[] = { t_q, t_kv, t_bt, t_su, nul, nul, cu, t_o };
    RadTensor* tarr = nullptr;
    if (hipHostMalloc((void**)&tarr, sizeof(host_arr), hipHostMallocMapped) != hipSuccess || !tarr) {
        printf("hipHostMalloc failed\n"); return 1;
    }
    memcpy(tarr, host_arr, sizeof(host_arr));

    RadParam pv[] = {
        RAD_INT("q_len", QLEN), RAD_INT("head_dim", HD), RAD_INT("gqa", GQA),
        RAD_INT("block_size", BS), RAD_INT("causal", 1), RAD_INT("window", 0),
        RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", "bf16"),
    };
    RadArgs a{}; a.p = pv; a.n_p = sizeof(pv) / sizeof(pv[0]);
    a.t = tarr; a.n_t = 8;
    a.scratch = dscratch; a.scratch_bytes = need;

    int rc = entry(&a, (RadStream)0);
    printf("entry rc=%d (QLEN=%d CTX=%d HD=%d SPL=%d need=%lld)\n", rc, QLEN, CTX, HD, SPL,
           (long long)need);
    hipDeviceSynchronize();
    if (rc != RAD_OK) return 2;

    std::vector<uint16_t> got(want.size());
    hipMemcpy(got.data(), dout, want.size() * 2, hipMemcpyDeviceToHost);
    std::vector<float> arena(need / 4);
    hipMemcpy(arena.data(), dscratch, need, hipMemcpyDeviceToHost);

    // ---- (a) device out vs oracle ----
    double num = 0, den = 0; int bad = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double e = bf16_f(got[i]) - bf16_f(want[i]);
        num += e * e; den += (double)bf16_f(want[i]) * bf16_f(want[i]);
        if (std::fabs(e) > 0.05) ++bad;
    }
    printf("(a) device  rel_l2=%.3e bad_cells=%d/%zu\n", std::sqrt(num / (den + 1e-30)), bad,
           want.size());

    // ---- (b) host re-merge of the arena ----
    // layout: so[idx*HD+d], sm[pn*SPL*HD + idx], sl[pn*SPL*(HD+1) + idx], idx=(t*QH+h)*SPL+sp
    const int64_t pn = (int64_t)QLEN * QH;
    const float* pso = arena.data();
    const float* psm = pso + pn * SPL * HD;
    const float* psl = psm + pn * SPL;
    int nan_slots = 0;
    for (int64_t i = 0; i < pn * SPL; ++i)
        if (std::isnan(psm[i]) || std::isnan(psl[i])) ++nan_slots;
    num = den = 0; bad = 0;
    for (int tok = 0; tok < QLEN; ++tok)
        for (int h = 0; h < QH; ++h) {
            float mx = -INFINITY;
            for (int sp = 0; sp < SPL; ++sp) { const float m = psm[((tok * QH + h) * SPL + sp)];
                if (m > mx) mx = m; }
            std::vector<float> acc(HD, 0.f); float denm = 0;
            for (int sp = 0; sp < SPL; ++sp) {
                const int64_t idx = (tok * QH + h) * SPL + sp;
                if (psm[idx] == -INFINITY) continue;
                const float w = exp2f(psm[idx] - mx);
                for (int dd = 0; dd < HD; ++dd) acc[dd] += w * pso[idx * HD + dd];
                denm += w * psl[idx];
            }
            for (int dd = 0; dd < HD; ++dd) {
                const float o = denm > 0 ? acc[dd] / denm : 0.f;
                const double e = o - (double)bf16_f(want[(size_t)tok * ROWW + h * HD + dd]);
                num += e * e; den += (double)bf16_f(want[(size_t)tok * ROWW + h * HD + dd])
                                    * bf16_f(want[(size_t)tok * ROWW + h * HD + dd]);
                if (std::fabs(e) > 0.05) ++bad;
            }
        }
    printf("(b) host-merge rel_l2=%.3e bad_cells=%d nan_slots=%d\n",
           std::sqrt(num / (den + 1e-30)), bad, nan_slots);

    // ---- m/l table, head 0 and head 1, all splits ----
    for (int h = 0; h < 2 && h < QH; ++h) {
        printf("h=%d:", h);
        for (int sp = 0; sp < SPL; ++sp) {
            const int64_t idx = h * SPL + sp;
            printf(" [%d m=%9.3f l=%9.3f]", sp, psm[idx], psl[idx]);
        }
        printf("\n");
    }
    return 0;
}
