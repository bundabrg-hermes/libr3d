// repro_attn_fp8.hip -- fp8-KV variant of repro_attn.hip, shaped after the fixture cases
// the WMMA tile fails (head_dim 128, gqa 4, causal 0, window 2048, q_len 1..8, per
// (seq,head) f32 descales, one unmapped block at the tail). The oracle is v1's own
// arithmetic chain: exact LUT decode (rad_fp8e4m3_to_f32), k_descale per key element,
// (float)exp(double) in ascending key order, v_descale per element -- i.e. libref's.
//
// Build/run (in radiance-dev on the card):
//   hipcc -O1 -std=c++20 -I/opt/radiance/include/radiance -w tools/repro_attn_fp8.hip \
//     -o /tmp/raf -ldl
//   LIBR3D_SO=$PWD/build3/radiance_home/kernels/libr3d.so /tmp/raf            # WMMA arm
//   R3D_NO_WMMA=1 ... /tmp/raf                                                 # v1 arm
// Exit 0 = arm within 2e-2 rel_l2 (v1 is expected bit-exact); nonzero otherwise.
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "rad_abi.h"
#include "rad_plugin.h"   /* rad_fp8e4m3_to_f32 -- the oracle's own decode */

static uint16_t rtne_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float bf16_f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static int envid(const char* k, int dflt) { const char* v = getenv(k); return v ? atoi(v) : dflt; }

int main() {
    const int HD = envid("REPRO_HD", 128);   /* Q62: 256 exercises the fp8 staging arms */
    const int BS = 16, NB = 4, QH = 8, KVH = 2, GQA = 4;
    const int QLEN = envid("REPRO_QLEN", 8);
    const int CTX  = envid("REPRO_CTX", 52);          // keys 48..51 land in unmapped block 3
    const int WIN  = envid("REPRO_WIN", 2048);
    const int NOP  = envid("REPRO_NOP", 0);           // 1 = descale operands absent
    const int DBG  = envid("REPRO_DBG", 0);           // 1 = kdes slot carries the dump buffer
    const int SLOTW = 2 * HD;                         // K then V per slot (fp8 bytes)
    const int ROWW = QH * HD;

    uint32_t st = 0x1357bd9fu;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };

    std::vector<uint16_t> hq((size_t)QLEN * ROWW);
    for (auto& v : hq) v = rtne_bf16(((rnd() & 0xffff) / 32768.0f) - 1.0f);
    // fp8 CODES straight into the plane; 0x7f/0xff (e4m3fn NaN) excluded, like any real
    // quantiser's output.
    std::vector<uint8_t> hkv((size_t)NB * KVH * BS * SLOTW);
    for (auto& v : hkv) { uint8_t b; do { b = (uint8_t)(rnd() & 0xff); } while (b == 0x7f || b == 0xff); v = b; }
    int32_t hbt[NB] = { 0, 1, 2, -1 };
    int32_t hsu[1]  = { CTX };
    float   hkdes[KVH], hvdes[KVH];
    for (int i = 0; i < KVH; ++i) { hkdes[i] = 0.002f + 0.004f * ((rnd() >> 3 & 63) / 63.0f);
                                    hvdes[i] = 0.010f + 0.020f * ((rnd() >> 5 & 63) / 63.0f); }

    const float scale = (float)(1.0 / sqrt((double)HD));
    // REPRO_ASU16=1: oracle loads the byte plane as uint16 pairs read as bf16 -- exactly
    // what a kernel with the bf16 branch misapplied to the fp8 plane would see. If an
    // arm's output matches THIS oracle, the arm is reading bf16 from the byte plane.
    const int ASU16 = envid("REPRO_ASU16", 0);
    const size_t plane_bytes = hkv.size();
    auto dec = [&](int64_t i) -> float {
        if (!ASU16) return rad_fp8e4m3_to_f32(hkv[(size_t)i]);
        if (2 * i + 1 >= (int64_t)plane_bytes) return 0.0f;
        return bf16_f((uint16_t)(hkv[(size_t)(2 * i)] | (hkv[(size_t)(2 * i + 1)] << 8)));
    };

    // ---- oracle: v1/libref's chain, element-order identical ----
    std::vector<uint16_t> want((size_t)QLEN * ROWW);
    for (int tok = 0; tok < QLEN; ++tok) {
        const int64_t pos = CTX - QLEN + tok;
        for (int h = 0; h < QH; ++h) {
            const int kh = h / GQA;
            const float kds = NOP ? 1.0f : hkdes[kh], vds = NOP ? 1.0f : hvdes[kh];
            float sc[1024]; float mx = -INFINITY; int nj = 0;
            for (int64_t j = 0; j < CTX; ++j) {
                const int64_t d = pos - j;
                if (WIN > 0 && (d >= WIN || d <= -WIN)) continue;
                const int blk = hbt[j / BS];
                if (blk < 0) continue;                       // unmapped: skipped, never zero-scored
                const int64_t kb = (int64_t)blk * KVH * BS * SLOTW + kh * BS * SLOTW + (j % BS) * SLOTW;
                float dot = 0;
                for (int dd = 0; dd < HD; ++dd)
                    dot += bf16_f(hq[(size_t)tok * ROWW + h * HD + dd])
                         * (dec(kb + dd) * kds);
                const float s = dot * scale;
                sc[nj++] = s; if (s > mx) mx = s;
            }
            float sum = 0, acc[HD] = {}; int jj = 0;
            for (int64_t j = 0; j < CTX; ++j) {
                const int64_t d = pos - j;
                if (WIN > 0 && (d >= WIN || d <= -WIN)) continue;
                const int blk = hbt[j / BS];
                if (blk < 0) continue;
                const int64_t kb = (int64_t)blk * KVH * BS * SLOTW + kh * BS * SLOTW + (j % BS) * SLOTW;
                const float w = (float)std::exp((double)(sc[jj++] - mx));
                sum += w;
                for (int dd = 0; dd < HD; ++dd)
                    acc[dd] += w * (dec(kb + HD + dd) * vds);
            }
            const float inv = sum > 0 ? 1.0f / sum : 0.0f;
            for (int dd = 0; dd < HD; ++dd)
                want[(size_t)tok * ROWW + h * HD + dd] = rtne_bf16(acc[dd] * inv);
        }
    }

    // ---- device + plugin ----
    void* dl = dlopen(getenv("LIBR3D_SO") ? getenv("LIBR3D_SO")
                                          : "/work/build3/radiance_home/kernels/libr3d.so", RTLD_NOW);
    if (!dl) { printf("dlopen: %s\n", dlerror()); return 1; }
    auto entry = (int (*)(const RadArgs*, RadStream))dlsym(dl, "r3d_rad_attn_paged");
    if (!entry) { printf("dlsym: %s\n", dlerror()); return 1; }

    uint16_t *dq, *dout; uint8_t *dkv; int32_t *dbt, *dsu; float *dkd, *dvd;
    hipMalloc(&dq, hq.size() * 2);   hipMemcpy(dq, hq.data(), hq.size() * 2, hipMemcpyHostToDevice);
    hipMalloc(&dkv, hkv.size());     hipMemcpy(dkv, hkv.data(), hkv.size(), hipMemcpyHostToDevice);
    hipMalloc(&dout, want.size() * 2); hipMemset(dout, 0, want.size() * 2);
    hipMalloc(&dbt, NB * 4);         hipMemcpy(dbt, hbt, NB * 4, hipMemcpyHostToDevice);
    hipMalloc(&dsu, 4);              hipMemcpy(dsu, hsu, 4, hipMemcpyHostToDevice);
    hipMalloc(&dkd, KVH * 4);        hipMemcpy(dkd, hkdes, KVH * 4, hipMemcpyHostToDevice);
    hipMalloc(&dvd, KVH * 4);        hipMemcpy(dvd, hvdes, KVH * 4, hipMemcpyHostToDevice);

    // REPRO_DBG=1: kdes slot carries a device double[64] dump buffer the R3D_DBG WMMA
    // build writes stage intermediates into (wg0 h0 s0). Pairs with NOP=1 semantics:
    // the kernel disables descales for the dump workgroup.
    double* ddbg = nullptr;
    if (DBG) {
        if (hipMalloc(&ddbg, 64 * 8) != hipSuccess) { printf("dbg malloc failed\n"); return 1; }
        hipMemset(ddbg, 0, 64 * 8);
    }

    RadTensor t_q{}, t_kv{}, t_bt{}, t_su{}, t_kd{}, t_vd{}, t_o{};
    t_q.data = dq; t_q.dtype = RAD_BF16; t_q.rank = 3;
    t_q.shape[0] = QLEN; t_q.shape[1] = QH; t_q.shape[2] = HD;
    t_q.stride[0] = ROWW; t_q.stride[1] = HD; t_q.stride[2] = 1;
    t_kv.data = dkv; t_kv.dtype = RAD_F8E4M3; t_kv.rank = 4;
    t_kv.shape[0] = NB; t_kv.shape[1] = KVH; t_kv.shape[2] = BS; t_kv.shape[3] = SLOTW;
    t_kv.stride[0] = (int64_t)KVH * BS * SLOTW; t_kv.stride[1] = (int64_t)BS * SLOTW;
    t_kv.stride[2] = SLOTW; t_kv.stride[3] = 1;
    t_bt.data = dbt; t_bt.dtype = RAD_I32; t_bt.rank = 2;
    t_bt.shape[0] = 1; t_bt.shape[1] = NB; t_bt.stride[0] = NB; t_bt.stride[1] = 1;
    t_su.data = dsu; t_su.dtype = RAD_I32; t_su.rank = 1; t_su.shape[0] = 1; t_su.stride[0] = 1;
    t_kd.data = dkd; t_kd.dtype = RAD_F32; t_kd.rank = 2; t_kd.shape[0] = 1; t_kd.shape[1] = KVH;
    t_kd.stride[0] = KVH; t_kd.stride[1] = 1;
    t_vd.data = dvd; t_vd.dtype = RAD_F32; t_vd.rank = 2; t_vd.shape[0] = 1; t_vd.shape[1] = KVH;
    t_vd.stride[0] = KVH; t_vd.stride[1] = 1;
    t_o.data = dout; t_o.dtype = RAD_BF16; t_o.rank = 3;
    t_o.shape[0] = QLEN; t_o.shape[1] = QH; t_o.shape[2] = HD;
    t_o.stride[0] = ROWW; t_o.stride[1] = HD; t_o.stride[2] = 1;
    // The kernels dereference descale RadTensor* ON DEVICE (ldt reads ->dtype), so the
    // tensor array must be GPU-addressable: pinned host memory is readable from both sides.
    RadTensor nulls[2] = {}, cu{}, t_dbg{};
    if (DBG) {
        t_dbg.data = ddbg; t_dbg.dtype = RAD_F32; t_dbg.rank = 1;
        t_dbg.shape[0] = 128; t_dbg.stride[0] = 1;                 /* 64 doubles of wire */
    }
    RadTensor host_arr[] = { t_q, t_kv, t_bt, t_su,
                             DBG ? t_dbg : (NOP ? nulls[0] : t_kd),
                             (DBG || NOP) ? nulls[1] : t_vd,
                             cu, t_o };
    RadTensor* tarr = nullptr;
    if (hipHostMalloc((void**)&tarr, sizeof(host_arr), hipHostMallocMapped) != hipSuccess || !tarr) {
        printf("hipHostMalloc failed\n"); return 1;
    }
    memcpy(tarr, host_arr, sizeof(host_arr));

    RadParam pv[] = {
        RAD_INT("q_len", QLEN), RAD_INT("head_dim", HD), RAD_INT("gqa", GQA),
        RAD_INT("block_size", BS), RAD_INT("causal", 0), RAD_INT("window", WIN),
        RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", "fp8_e4m3"),
    };
    RadArgs a{}; a.p = pv; a.n_p = sizeof(pv) / sizeof(pv[0]);
    a.t = tarr; a.n_t = 8;

    int rc = entry(&a, (RadStream)0);
    printf("entry rc=%d  (QLEN=%d CTX=%d WIN=%d NOP=%d descale k=[%.4f %.4f])\n",
           rc, QLEN, CTX, WIN, NOP, hkdes[0], hkdes[1]);
    hipDeviceSynchronize();
    if (DBG && ddbg) {
        double hd[64];
        hipMemcpy(hd, ddbg, sizeof(hd), hipMemcpyDeviceToHost);
        printf("dump rb=%.0f rawK=%.0f uK=0x%04x | S0=%.4e S16=%.4e m0=%.4f l0=%.4f scale2=%.6f\n",
               hd[0], hd[1], (unsigned)hd[2], hd[3], hd[4], hd[5], hd[6], hd[7]);
        printf("dump2 S0=%.4e P0=%.4f m=%.4f | aV=%.0f rawV=%.0f uV=0x%04x | ll=%.6g O=%.6g o=%.6g vds=%.4g\n",
               hd[8], hd[9], hd[10], hd[11], hd[12], (unsigned)hd[13], hd[14], hd[15], hd[16], hd[17]);
        printf("fa.x:");  for (int i = 0; i < 8; ++i) printf(" %.4g", hd[18 + i]);  printf("\n");
        printf("fb.x:");  for (int i = 0; i < 8; ++i) printf(" %.4g", hd[26 + i]);  printf("\n");
        printf("acc.x:"); for (int i = 0; i < 8; ++i) printf(" %.4g", hd[42 + i]);  printf("\n");
        printf("KVp[0]=%.4g KVp[1]=%.4g Qsp[0]=%.4g Qsp[1]=%.4g\n", hd[50], hd[51], hd[52], hd[53]);
        printf("fresh: Qsp0=%.4g KVp0=%.4g Qsp0_again=%.4g\n", hd[54], hd[55], hd[56]);
        printf("raw bits: Qsp0=0x%04x  qplane[qbase]=0x%04x  KVp0=0x%04x Qsp0b=0x%04x\n",
               (unsigned)hd[57], (unsigned)hd[58], (unsigned)hd[59], (unsigned)hd[60]);
        printf("host truth: q[0]=0x%04x (%.4g) kvbyte[0]=%u -> decode %.4g\n",
               hq[0], bf16_f(hq[0]), hkv[0], rad_fp8e4m3_to_f32(hkv[0]));
    }

    std::vector<uint16_t> got(want.size());
    hipMemcpy(got.data(), dout, want.size() * 2, hipMemcpyDeviceToHost);

    double num = 0, den = 0, max_abs = 0; int exact = 0, nan = 0, shown = 0;
    for (size_t e = 0; e < want.size(); ++e) {
        float w = bf16_f(want[e]), g = bf16_f(got[e]);
        if (!std::isfinite(g)) ++nan;
        if (want[e] == got[e]) ++exact;
        else if (shown < 6) {
            printf("  diff e=%zu tok=%zu h=%zu d=%zu want %.4f (0x%04x) got %.4g (0x%04x)\n",
                   e, e / ROWW, (e % ROWW) / HD, e % HD, w, want[e], g, got[e]);
            ++shown;
        }
        num += (double)(g - w) * (g - w);
        den += (double)w * w;
        if (fabs(g - w) > max_abs) max_abs = fabs(g - w);
    }
    double rel = den > 0 ? sqrt(num / den) : 0;
    printf("QLEN=%d: exact %zu/%zu  rel_l2=%.3e max_abs=%.3e  nonfinite=%d\n",
           QLEN, (size_t)exact, want.size(), rel, max_abs, nan);
    return (rc == 0 && nan == 0 && rel < 2e-2) ? 0 : 2;
}
