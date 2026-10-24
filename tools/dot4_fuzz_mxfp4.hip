// dot4_fuzz_mxfp4.hip -- fuzz the mxfp4 dot4 q8x arm's constants and conventions
// against INDEPENDENT decoders (the probe-era trap: an LUT verified against a copy
// of itself passes every uniform probe and fails on real data). Three checks:
//
//   1. R3D_E2M1_X2 (src/r3d_mxfp4_q8x.h -- the shipped repack LUT) vs the ABI's own
//      rad_fp4e2m1_to_f32, over ALL 16 codes including 7/15. The probe's "codes
//      7/15 are NaN, never generated" law was the table-vs-copy-of-itself artifact:
//      its generator excluded them AND its f64 model shared its table. OCP e2m1 has
//      NO NaN: code 7 is +6.0 (x2 = 12), code 15 is -6.0.
//   2. The integrated (signed,signed) sudot4 spelling -- builtin form
//      __builtin_amdgcn_sudot4(1,a,1,b,acc,0), the gemv's exact call -- over random
//      dwords dense with negatives and byte-boundary patterns, vs a CPU model.
//   3. r3d_e8m0_sc executed ON THE DEVICE over all 256 scale bytes vs the oracle's
//      e8m0 rules (0xFF NaN, 0x00 subnormal 2^-127, else the biased pattern).
//
// Build (on cat, inside radiance-dev:v1.11.0):
//   hipcc -O2 -ffp-contract=off --offload-arch=gfx1100 \
//         -I<repo>/src -I/opt/radiance/include/radiance tools/dot4_fuzz_mxfp4.hip -o dot4_fuzz_mxfp4
// Run with the gfx1100 index pinned (ROCR_VISIBLE_DEVICES from enumeration -- the
// tool aborts unless the device it sees is a gfx1100).

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>

#include "rad_plugin.h"        // rad_fp4e2m1_to_f32 -- the independent decoder
#include "r3d_mxfp4_q8x.h"     // the shipped constants under test

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_)); exit(2); } } while (0)

// ---- device fuzzers ---------------------------------------------------------------

__global__ void dot4_fuzz(const int32_t* a, const int32_t* b, int32_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __builtin_amdgcn_sudot4(1, a[i], 1, b[i], 0x5a5a5a5a, 0);
}

__global__ void e8m0_fuzz(uint32_t* out) {         // f32 bits of r3d_e8m0_sc(byte)
    const int i = threadIdx.x;
    if (i < 256) {
        const float f = r3d_e8m0_sc((uint8_t)i);
        uint32_t u;
        memcpy(&u, &f, 4);
        out[i] = u;
    }
}

__global__ void repack_pair_fuzz(uint16_t* out) {  // the repack's byte -> pair form
    constexpr int8_t lut[16] = R3D_E2M1_X2;
    const int b = threadIdx.x;
    if (b < 256)
        out[b] = (uint16_t)(uint8_t)lut[b & 15u] | ((uint16_t)(uint8_t)lut[b >> 4] << 8);
}

// CPU model of signed s8 x s8 dot4 with wraparound accumulate (probe's dot4_fuzz.hip)
static int32_t dot(int32_t a, int32_t b) {
    int32_t acc = 0x5a5a5a5a;
    for (int j = 0; j < 4; j++) {
        const int32_t av = (int32_t)(int8_t)((uint32_t)a >> (8 * j));
        const int32_t bv = (int32_t)(int8_t)((uint32_t)b >> (8 * j));
        acc = (int32_t)((uint64_t)acc + (uint64_t)(int64_t)((int64_t)av * bv));
    }
    return acc;
}

struct Rng { uint64_t s; Rng(uint64_t seed):s(seed|1){}
    uint32_t u32(){ s ^= s<<13; s ^= s>>7; s ^= s<<17; return (uint32_t)(s>>32); }
    uint32_t below(uint32_t n){ return u32() % n; } };

int main() {
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, 0));
    printf("DEV %s (%s)\n", prop.name, prop.gcnArchName);
    if (!strstr(prop.gcnArchName, "gfx1100") && !strstr(prop.gcnArchName, "gfx1101") &&
        !strstr(prop.gcnArchName, "gfx1102") && !strstr(prop.gcnArchName, "gfx1103")) {
        fprintf(stderr, "ABORT: not a gfx11 card\n"); return 3; }
    int rc = 0;
    Rng rng(0xD074);

    // ---- 1. the repack LUT vs the ABI decoder, all 16 codes, randomized order ----
    long lut_bad = 0;
    for (int rep = 0; rep < 4096; rep++) {
        const unsigned c = rng.below(16);
        const float v = rad_fp4e2m1_to_f32((uint8_t)c);
        const int8_t want = (int8_t)lrintf(2.0f * v);
        constexpr int8_t lut[16] = R3D_E2M1_X2;
        if (lut[c] != want) {
            if (lut_bad < 8) printf("LUT code=%u want=%d got=%d\n", c, want, lut[c]);
            lut_bad++;
        }
    }
    // every byte value: the pair the repack writes vs 2x both ABI-decoded codes
    for (unsigned b = 0; b < 256; b++) {
        constexpr int8_t lut[16] = R3D_E2M1_X2;
        const int8_t wlo = (int8_t)lrintf(2.0f * rad_fp4e2m1_to_f32((uint8_t)(b & 15u)));
        const int8_t whi = (int8_t)lrintf(2.0f * rad_fp4e2m1_to_f32((uint8_t)(b >> 4)));
        const uint16_t want = (uint16_t)(uint8_t)wlo | ((uint16_t)(uint8_t)whi << 8);
        const uint16_t got = (uint16_t)(uint8_t)lut[b & 15u] | ((uint16_t)(uint8_t)lut[b >> 4] << 8);
        if (got != want) { if (lut_bad < 16) printf("PAIR byte=%02x want=%04x got=%04x\n", b, want, got); lut_bad++; }
    }
    printf("FUZZ1 e2m1x2 LUT vs rad_fp4e2m1_to_f32 (all 16 codes + 256 byte pairs): %ld bad\n", lut_bad);
    if (lut_bad) rc = 1;

    // ---- 2. device: the integrated signed dot4 spelling ----
    const int n = 1 << 20;
    int32_t *ha = new int32_t[n], *hb = new int32_t[n];
    for (int i = 0; i < n; i++) {
        ha[i] = (int32_t)rng.u32();
        hb[i] = (int32_t)rng.u32();
    }
    for (int i = 0; i < n; i += 7) { ha[i] = (int32_t)0x807FFF00u; hb[i] = (int32_t)0xFF01807Fu; }
    int32_t *d_a, *d_b, *d_o;
    CK(hipMalloc((void**)&d_a, n * 4)); CK(hipMalloc((void**)&d_b, n * 4));
    CK(hipMalloc((void**)&d_o, n * 4));
    CK(hipMemcpy(d_a, ha, n * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_b, hb, n * 4, hipMemcpyHostToDevice));
    dot4_fuzz<<<(n + 255) / 256, 256>>>(d_a, d_b, d_o, n);
    CK(hipGetLastError());
    // device e8m0 + repack-pair sweeps
    uint32_t* d_e; uint16_t* d_p;
    CK(hipMalloc((void**)&d_e, 256 * 4)); CK(hipMalloc((void**)&d_p, 256 * 2));
    e8m0_fuzz<<<1, 256>>>(d_e);
    repack_pair_fuzz<<<1, 256>>>(d_p);
    CK(hipGetLastError()); CK(hipDeviceSynchronize());
    int32_t* ho = new int32_t[n];
    CK(hipMemcpy(ho, d_o, n * 4, hipMemcpyDeviceToHost));
    long dot_bad = 0;
    for (int i = 0; i < n; i++)
        if (ho[i] != dot(ha[i], hb[i])) {
            if (dot_bad < 3) printf("DOT i=%d a=%08x b=%08x gpu=%08x cpu=%08x\n",
                                    i, (uint32_t)ha[i], (uint32_t)hb[i], (uint32_t)ho[i], (uint32_t)dot(ha[i], hb[i]));
            dot_bad++;
        }
    printf("FUZZ2 sudot4(1,a,1,b,acc,0) s8xs8: %ld/%d bad\n", dot_bad, n);
    if (dot_bad) rc = 1;

    // ---- 3. device e8m0 decode vs oracle rules; device repack pairs vs LUT ----
    uint32_t he[256]; uint16_t hp[256];
    CK(hipMemcpy(he, d_e, 256 * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hp, d_p, 256 * 2, hipMemcpyDeviceToHost));
    long e_bad = 0, p_bad = 0;
    for (unsigned b = 0; b < 256; b++) {
        uint32_t want;
        if (b == 0xffu) want = 0x7fc00000u;
        else if (b == 0x00u) want = 0x00400000u;                 // 2^-127 subnormal
        else want = (uint32_t)b << 23;
        const bool ok = (b == 0xffu) ? (he[b] & 0x7f800000u) == 0x7f800000u && (he[b] & 0x007fffffu)
                                     : he[b] == want;            // any NaN payload accepted
        if (!ok) { if (e_bad < 4) printf("E8M0 b=%02x want=%08x got=%08x\n", b, want, he[b]); e_bad++; }
        constexpr int8_t lut[16] = R3D_E2M1_X2;
        const uint16_t wp = (uint16_t)(uint8_t)lut[b & 15u] | ((uint16_t)(uint8_t)lut[b >> 4] << 8);
        if (hp[b] != wp) { if (p_bad < 4) printf("PAIRDEV b=%02x want=%04x got=%04x\n", b, wp, hp[b]); p_bad++; }
    }
    printf("FUZZ3 device e8m0 decode: %ld/256 bad | device repack pairs: %ld/256 bad\n", e_bad, p_bad);
    if (e_bad || p_bad) rc = 1;

    printf("%s\n", rc ? "DOT4 FUZZ: FAIL" : "DOT4 FUZZ: OK");
    return rc;
}
