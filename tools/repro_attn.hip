// repro_attn.hip -- standalone reproduction harness for the attn_paged kbench failure.
// Builds one small hand-checkable geometry (n_seq=1, gqa=1, 2 mapped blocks + 1 unmapped,
// ctx spanning a block boundary, causal, bf16 cache), calls r3d_rad_attn_paged through the
// plugin .so, and compares element by element against libref's arithmetic run on the host.
// Build/run (in radiance-dev on the card):
//   hipcc -O1 -std=c++20 -I/opt/radiance/include/radiance -w tools/repro_attn.hip \
//     -o /tmp/ra -ldl
//   LIBR3D_SO=$PWD/build3/radiance_home/kernels/libr3d.so /tmp/ra
// The pass condition is 0 differing elements (exp() rounding may cost one bf16-ULP
// straggler -- shown, not hidden).
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

int main() {
    const int HD = 8, BS = 4, NB = 3, QLEN = 2, QH = 1, KVH = 1, GQA = 1;
    const int CTX = 5;                       // keys 0..4: block 0 slots 0-3, block 1 slot 0
    const int SLOTW = 2 * HD;                // K then V per slot
    const int ROWW = QH * HD;

    uint32_t st = 0x51ed2701u;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };

    std::vector<uint16_t> hq((size_t)QLEN * ROWW), hkv((size_t)NB * KVH * BS * SLOTW);
    for (auto& v : hq)  v = rtne_bf16(((rnd() & 0xffff) / 32768.0f) - 1.0f);
    for (auto& v : hkv) v = rtne_bf16(((rnd() & 0xffff) / 32768.0f) - 1.0f);
    int32_t hbt[NB] = { 0, 1, -1 };          // third block unmapped: keys 8..11 unreachable
    int32_t hsu[1]  = { CTX };

    // ---- host oracle, ref_attn_paged's chain ----
    const float scale = 1.0f / sqrtf((float)HD);
    std::vector<uint16_t> want((size_t)QLEN * ROWW);
    for (int tok = 0; tok < QLEN; ++tok) {
        const int i = tok, pos = CTX - QLEN + i;
        std::vector<float> sc;
        float mx = -INFINITY;
        for (int j = 0; j < CTX; ++j) {      // causal only, every key visible <= pos
            if (j > pos) continue;
            const int blk = hbt[j / BS];
            const int64_t kb = (int64_t)blk * KVH * BS * SLOTW + (j % BS) * SLOTW;
            float dot = 0;
            for (int d = 0; d < HD; ++d)
                dot += bf16_f(hq[(size_t)tok * ROWW + d]) * (bf16_f(hkv[kb + d]) * 1.0f);
            const float s = dot * scale;
            sc.push_back(s);
            if (s > mx) mx = s;
        }
        float sum = 0, acc[HD] = {};
        int jj = 0;
        for (int j = 0; j <= pos; ++j, ++jj) {
            const int blk = hbt[j / BS];
            const int64_t kb = (int64_t)blk * KVH * BS * SLOTW + (j % BS) * SLOTW;
            const float w = (float)std::exp((double)(sc[jj] - mx));
            sum += w;
            for (int d = 0; d < HD; ++d)
                acc[d] += w * (bf16_f(hkv[kb + HD + d]) * 1.0f);
        }
        const float inv = sum > 0 ? 1.0f / sum : 0.0f;
        for (int d = 0; d < HD; ++d) want[(size_t)tok * ROWW + d] = rtne_bf16(acc[d] * inv);
    }

    // ---- device + plugin call ----
    void* dl = dlopen(getenv("LIBR3D_SO") ? getenv("LIBR3D_SO")
                                          : "/work/build3/radiance_home/kernels/libr3d.so", RTLD_NOW);
    if (!dl) { printf("dlopen: %s\n", dlerror()); return 1; }
    auto entry = (int (*)(const RadArgs*, RadStream))dlsym(dl, "r3d_rad_attn_paged");
    if (!entry) { printf("dlsym: %s\n", dlerror()); return 1; }

    uint16_t *dq, *dkv, *dout; int32_t *dbt, *dsu;
    hipMalloc(&dq,  hq.size() * 2);   hipMemcpy(dq,  hq.data(),  hq.size() * 2,  hipMemcpyHostToDevice);
    hipMalloc(&dkv, hkv.size() * 2);  hipMemcpy(dkv, hkv.data(), hkv.size() * 2, hipMemcpyHostToDevice);
    hipMalloc(&dout, want.size() * 2); hipMemset(dout, 0, want.size() * 2);
    hipMalloc(&dbt, NB * 4);          hipMemcpy(dbt, hbt, NB * 4, hipMemcpyHostToDevice);
    hipMalloc(&dsu, 4);               hipMemcpy(dsu, hsu, 4, hipMemcpyHostToDevice);

    RadTensor t_q{}, t_kv{}, t_bt{}, t_su{}, t_o{};
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
    RadTensor nulls[3] = {};                       // k_descale, v_descale, cu_seqlens absent
    RadTensor tensors[] = { t_q, t_kv, t_bt, t_su, nulls[0], nulls[1], nulls[2], t_o };

    RadParam pv[] = {
        RAD_INT("q_len", QLEN), RAD_INT("head_dim", HD), RAD_INT("gqa", GQA),
        RAD_INT("block_size", BS), RAD_INT("causal", 1), RAD_INT("window", 0),
        RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", "bf16"),
    };
    RadArgs a{}; a.p = pv; a.n_p = sizeof(pv) / sizeof(pv[0]);
    a.t = tensors; a.n_t = 8;

    int rc = entry(&a, (RadStream)0);
    printf("entry rc=%d\n", rc);
    hipDeviceSynchronize();

    std::vector<uint16_t> got(want.size());
    hipMemcpy(got.data(), dout, want.size() * 2, hipMemcpyDeviceToHost);

    int bad = 0;
    for (size_t e = 0; e < want.size(); ++e)
        if (want[e] != got[e]) {
            if (bad < 16) printf("tok=%zu d=%zu want %.6f got %.6f\n",
                                 e / ROWW, e % ROWW, bf16_f(want[e]), bf16_f(got[e]));
            ++bad;
        }
    printf("QLEN=%d CTX=%d: %d/%zu elements differ\n", QLEN, CTX, bad, want.size());
    return bad ? 2 : 0;
}
