// bench_dot4_seed.hip -- VALIDATED PROBE SEED (hipfire Phase B, 2026-10-24), kept
// verbatim below for provenance of the integrated arm in src/r3d_gemm_mxfp4.hip.
// ONE LAW BELOW IS WRONG AND THE INTEGRATION FIXES IT: "codes 7/15 (NaN) never
// generated" -- OCP e2m1 has NO NaN (rad_plugin.h:356: "all sixteen codes are
// numbers, max 6"; code 7 = +6.0). The probe's parity was blind to this because
// its generator excluded 7/15 AND its f64 model shared kE2M1X2 -- the table-vs-
// copy-of-itself trap. The shipped arm's table is R3D_E2M1_X2 in
// src/r3d_mxfp4_q8x.h (7 -> 12, 15 -> -12), fuzzed against the ABI decoder by
// tools/dot4_fuzz_mxfp4.hip. Do not reuse kE2M1X2 below for anything new.
//
// bench_dot4.hip — dot4-transfer probe for libr3d's mxfp4 decode geometry. Standalone.
//
// ARM B: repacked byte weight plane (one signed i8 = e2m1 x 2 per code, pitch K)
//        + the SHARED separate e8m0 scale plane (u8 per 32, row-major)
//        + i8 activations fed PRE-REQUANTIZED (requant is a separate, time-excluded
//          prologue kernel: per 32-group max -> s_g, r_k = rintf(e4m3/s_g)),
//        + pure signed v_dot4 (s8 x s8, fuzzed 2026-10-24), hipfire geometry: one
//          warp per weight row in a wave32 block, lane owns 8 codes (2 dot4), quad
//          owns the 32-group, per-group int32 flush ((float)sub * s_g * AS * 2^(bs-127) * 0.5),
//          warp butterfly of the 8 group terms, PREF-deep weight+scale ring.
// ARM C: ARM B's math + hipfire's interleaved single stream: 136B groups per 128
//        weights: [0..3] = the four e8m0 scale bytes, [4..7] = pad, [8..135] = 128 i8
//        weights. One stream carries weights AND scales; the reader never touches a
//        second plane.
//
// Parity: per-arm f64/f32 CPU model of ITS OWN chain (requant included), deterministic
// enough to be BIT-EXACT against the bf16 output: integer sub, f32 flush in the
// kernel's exact multiply order, the exact shuffle butterfly tree, per-step serial
// accumulation, bf16 RNE store. Y is poisoned 0xFF first; untouched outputs read NaN.
//
// usage: bench_dot4 parity  <arm> <N> <K> <M>
//        bench_dot4 time    <arm> <M> <K> <N> <reps> <rot>
//        bench_dot4 prologue <M> <K> <reps>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <vector>

__device__ __host__ static inline float  bits2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
__device__ __host__ static inline uint32_t f2bits(float f)  { uint32_t u; memcpy(&u, &f, 4); return u; }

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_)); exit(2); } } while (0)

// ---------- decoders (ports of the shipped ones; host+device) ----------------------

__device__ __host__ inline float e4m3_dec(uint8_t b) {   // r3d_gemm_mxfp4 fp8e4m3_gemv
    const uint32_t s = (uint32_t)(b >> 7) << 31;
    const uint32_t e = (uint32_t)(b >> 3) & 0xfu, m = (uint32_t)b & 0x7u;
    uint32_t u;
    if (e == 0u) {
        if (m == 0u) return bits2f(s);
        const float f = (float)m / 512.0f;
        u = s | (f2bits(f) & 0x7fffffffu);
    } else if (e == 0xfu && m == 0x7u) {
        u = s | 0x7fc00000u;
    } else {
        u = s | ((e - 7u + 127u) << 23) | (m << 20);
    }
    return bits2f(u);
}

__device__ __host__ inline uint16_t bf16_rne(float f) {  // r3d f32_to_bf16_m4
    const uint32_t u = f2bits(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

static const int kE2M1X2[16] = { 0,1,2,3,4,6,8, 0,  0,-1,-2,-3,-4,-6,-8, 0 };
// probe law: codes 7/15 (NaN) never generated; repack asserts them absent.

static inline double now_s() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// fast random bytes (never memset/zeros — project law)
struct Rng { uint64_t s; Rng(uint64_t seed):s(seed|1){}
    uint32_t u32(){ s ^= s<<13; s ^= s>>7; s ^= s<<17; return (uint32_t)(s>>32); }
    uint32_t below(uint32_t n){ return u32() % n; } };

// ---------- kernels ----------------------------------------------------------------

static const int kWave = 32;

// Prologue: e4m3 codes + per-32 group -> i8 activations + per-group f32 requant scale.
// (AS is NOT folded here; it rides the flush, per the probe spec.)
// warp-per-group so K=4096 spreads over many CUs (the thread-per-group draft sat at a
// 11-42 us latency floor on 1-4 blocks; this is what integration would ship).
__global__ __launch_bounds__(256) void prologue_requant(
    const uint8_t* __restrict__ A, int8_t* __restrict__ X8, float* __restrict__ SG,
    int M, int K)
{
    const int cols = K >> 5;                       // groups per row
    const int gwarp = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (gwarp >= M * cols) return;
    const int m = gwarp / cols, g = gwarp - m * cols;
    const int lane = threadIdx.x & 31;
    const float e = e4m3_dec(A[(size_t)m * K + (g << 5) + lane]);
    float mx = fabsf(e);
#pragma unroll
    for (int off = 16; off; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off));  // max is order-free
    const float s = mx ? mx / 127.0f : 1.0f;
    if (lane == 0) SG[(size_t)m * cols + g] = s;
    X8[(size_t)m * K + (g << 5) + lane] = (int8_t)rintf(e / s);
}

// ARM A (stretch). SHIPPED nibble layout (K/2 bytes/row, no repack) + the separate
// e8m0 BS plane; an LDS LUT turns each code BYTE (two codes) into a dword
// [v_lo, v_hi, v_lo, v_hi] (i8 bits, sign baked in), two v_prmt merges pack the
// lane's 8 codes into two i8 dwords, then the ARM B dot4/flush chain verbatim.
// LUT law: negatives are written as their own literals, parity on random codes over
// all 256 byte values IS the independent-decoder fuzz for this table.
template <int MT, int PREF>
__global__ __launch_bounds__(32) void gemv_A(
    const uint8_t* __restrict__ NIB, const uint8_t* __restrict__ BS,
    const int8_t* __restrict__ X, const float* __restrict__ SG, const float* __restrict__ AS,
    uint16_t* __restrict__ Y, int N, int K)
{
    __shared__ uint32_t LUT[256];
    {
        for (int b = threadIdx.x; b < 256; b += 32) {
            const int cl = b & 15, ch = b >> 4;
            const uint32_t vl = (uint32_t)(uint8_t)(int8_t)kE2M1X2[cl];
            const uint32_t vh = (uint32_t)(uint8_t)(int8_t)kE2M1X2[ch];
            LUT[b] = vl | (vh << 8) | (vl << 16) | (vh << 24);
        }
    }
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    const int q = t >> 2;
    const int nstep  = K >> 8;
    const int bs_col = K >> 5;
    const int as_col = K >> 7;
    const uint8_t* crow = NIB + (size_t)r * (K >> 1);
    const uint8_t* brow = BS + (size_t)r * bs_col;

    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; m++) acc[m] = 0.0f;
    __syncthreads();                             // LUT visible

    uint32_t w = 0u; uint8_t sb = 0;
    if constexpr (PREF == 2) { w = *(const uint32_t*)(crow + (t << 2)); sb = brow[q]; }

    for (int it = 0; it < nstep; it++) {
        uint32_t wn = 0u; uint8_t sbn = 0;
        if constexpr (PREF == 2) {
            if (it + 1 < nstep) {
                wn = *(const uint32_t*)(crow + ((it + 1) << 7) + (t << 2));
                sbn = brow[((it + 1) << 3) + q];
            }
        }
        const uint32_t bc = (PREF == 2) ? w : *(const uint32_t*)(crow + (it << 7) + (t << 2));
        const uint8_t sc_b = (PREF == 2) ? sb : brow[(it << 3) + q];
        const uint32_t a0 = LUT[bc & 0xffu];
        const uint32_t a1 = LUT[(bc >> 8) & 0xffu];
        const uint32_t a2 = LUT[(bc >> 16) & 0xffu];
        const uint32_t a3 = LUT[bc >> 24];
        const uint32_t wa = __byte_perm(a0, a1, 0x5410);   // [b0lo,b0hi,b1lo,b1hi]
        const uint32_t wb = __byte_perm(a2, a3, 0x5410);
        const int g  = (it << 3) + q;
        const float sc = bits2f((uint32_t)sc_b << 23);
#pragma unroll
        for (int m = 0; m < MT; m++) {
            const uint2 xv = *(const uint2*)(X + (size_t)m * K + (it << 8) + (t << 3));
            int32_t p = __builtin_amdgcn_sudot4(1, (int)xv.x, 1, (int)wa, 0, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xv.y, 1, (int)wb, p, 0);
            p += __shfl_xor(p, 1);
            p += __shfl_xor(p, 2);
            float c = (float)p * SG[(size_t)m * bs_col + g] * AS[(size_t)m * as_col + (g >> 2)] * sc * 0.5f;
            c += __shfl_xor(c, 4);
            c += __shfl_xor(c, 8);
            c += __shfl_xor(c, 16);
            acc[m] += c;
        }
        if constexpr (PREF == 2) { w = wn; sb = sbn; }
    }
    if (t == 0) {
#pragma unroll
        for (int m = 0; m < MT; m++) Y[(size_t)m * N + r] = bf16_rne(acc[m]);
    }
}

// ARM B. W = signed i8 plane [N][K]; BS = e8m0 [N][K/32]; X = i8 [MT][K];
// SG = f32 [MT][K/32]; AS = f32 [MT][K/128]; Y = bf16 [MT][N].
template <int MT, int PREF>
__global__ __launch_bounds__(32) void gemv_B(
    const int8_t* __restrict__ W, const uint8_t* __restrict__ BS,
    const int8_t* __restrict__ X, const float* __restrict__ SG, const float* __restrict__ AS,
    uint16_t* __restrict__ Y, int N, int K)
{
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    const int q = t >> 2;
    const int nstep  = K >> 8;
    const int bs_col = K >> 5;
    const int as_col = K >> 7;
    const int8_t*  crow = W  + (size_t)r * K;
    const uint8_t* brow = BS + (size_t)r * bs_col;

    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; m++) acc[m] = 0.0f;

    uint2 w = make_uint2(0u, 0u); uint8_t sb = 0;
    if constexpr (PREF == 2) { w = *(const uint2*)(crow + (t << 3)); sb = brow[q]; }

    for (int it = 0; it < nstep; it++) {
        uint2 wn = make_uint2(0u, 0u); uint8_t sbn = 0;
        if constexpr (PREF == 2) {
            if (it + 1 < nstep) {
                wn = *(const uint2*)(crow + ((it + 1) << 8) + (t << 3));
                sbn = brow[((it + 1) << 3) + q];
            }
        }
        const uint2  wc = (PREF == 2) ? w  : *(const uint2*)(crow + (it << 8) + (t << 3));
        const uint8_t sc_b = (PREF == 2) ? sb : brow[(it << 3) + q];
        const int g  = (it << 3) + q;          // quad owns the 32-group
        const float sc = bits2f((uint32_t)sc_b << 23);
#pragma unroll
        for (int m = 0; m < MT; m++) {
            const uint2 xv = *(const uint2*)(X + (size_t)m * K + (it << 8) + (t << 3));
            int32_t p = __builtin_amdgcn_sudot4(1, (int)xv.x, 1, (int)wc.x, 0, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xv.y, 1, (int)wc.y, p, 0);
            p += __shfl_xor(p, 1);
            p += __shfl_xor(p, 2);
            float c = (float)p * SG[(size_t)m * bs_col + g] * AS[(size_t)m * as_col + (g >> 2)] * sc * 0.5f;
            c += __shfl_xor(c, 4);
            c += __shfl_xor(c, 8);
            c += __shfl_xor(c, 16);
            acc[m] += c;
        }
        if constexpr (PREF == 2) { w = wn; sb = sbn; }
    }
    if (t == 0) {
#pragma unroll
        for (int m = 0; m < MT; m++) Y[(size_t)m * N + r] = bf16_rne(acc[m]);
    }
}

// ARM C. S = interleaved plane: row pitch K + 8*(K>>7); 136B groups (4 scales, 4 pad, 128 i8).
template <int MT, int PREF>
__global__ __launch_bounds__(32) void gemv_C(
    const uint8_t* __restrict__ S,
    const int8_t* __restrict__ X, const float* __restrict__ SG, const float* __restrict__ AS,
    uint16_t* __restrict__ Y, int N, int K)
{
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    const int q = t >> 2;
    const int nstep  = K >> 8;
    const int bs_col = K >> 5;
    const int as_col = K >> 7;
    const uint8_t* crow = S + (size_t)r * (size_t)(K + ((K >> 7) << 3));

    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; m++) acc[m] = 0.0f;

    const int hi8 = (t >> 4) * 136;            // second 136B group of the step
    const int lo8 = (t & 15) << 3;

    uint2 w = make_uint2(0u, 0u); uint8_t sb = 0;
    if constexpr (PREF == 2) {
        w = *(const uint2*)(crow + hi8 + 8 + lo8);
        sb = crow[((q >> 2) * 136) + (q & 3)];
    }

    for (int it = 0; it < nstep; it++) {
        uint2 wn = make_uint2(0u, 0u); uint8_t sbn = 0;
        if constexpr (PREF == 2) {
            if (it + 1 < nstep) {
                const uint8_t* np = crow + (((it + 1) << 1) * 136);
                wn = *(const uint2*)(np + hi8 + 8 + lo8);
                sbn = np[((q >> 2) * 136) + (q & 3)];
            }
        }
        const uint8_t* sp = crow + ((it << 1) * 136);
        const uint2   wc = (PREF == 2) ? w : *(const uint2*)(sp + hi8 + 8 + lo8);
        const uint8_t sc_b = (PREF == 2) ? sb : sp[((q >> 2) * 136) + (q & 3)];
        const int g  = (it << 3) + q;
        const float sc = bits2f((uint32_t)sc_b << 23);
#pragma unroll
        for (int m = 0; m < MT; m++) {
            const uint2 xv = *(const uint2*)(X + (size_t)m * K + (it << 8) + (t << 3));
            int32_t p = __builtin_amdgcn_sudot4(1, (int)xv.x, 1, (int)wc.x, 0, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xv.y, 1, (int)wc.y, p, 0);
            p += __shfl_xor(p, 1);
            p += __shfl_xor(p, 2);
            float c = (float)p * SG[(size_t)m * bs_col + g] * AS[(size_t)m * as_col + (g >> 2)] * sc * 0.5f;
            c += __shfl_xor(c, 4);
            c += __shfl_xor(c, 8);
            c += __shfl_xor(c, 16);
            acc[m] += c;
        }
        if constexpr (PREF == 2) { w = wn; sb = sbn; }
    }
    if (t == 0) {
#pragma unroll
        for (int m = 0; m < MT; m++) Y[(size_t)m * N + r] = bf16_rne(acc[m]);
    }
}

// ---------- operand generation + repacks (host) -------------------------------------

struct Op {
    int N, K, M;
    std::vector<uint8_t> A;     // [M][K] e4m3 codes
    std::vector<float>   AS;    // [M][K/128]
    std::vector<uint8_t> NIB;   // [N][K/2] nibble codes (the shipped layout)
    std::vector<uint8_t> BS;    // [N][K/32] e8m0
    std::vector<int8_t>  W8;    // [N][K] repacked (e2m1 x2, signed)
    std::vector<uint8_t> S136;  // [N][K + 8*K/128] interleaved
};

static void gen_ops(Op& o, uint64_t seed, bool want_w8, bool want_s136, bool want_nib) {
    Rng r(seed);
    const int K32 = o.K >> 5, K128 = o.K >> 7;
    o.A.resize((size_t)o.M * o.K);
    o.AS.assign((size_t)o.M * K128, 0.f);
    o.NIB.resize((size_t)o.N * (o.K >> 1));
    o.BS.resize((size_t)o.N * K32);
    // 12 valid e2m1 codes (7/15 excluded by law)
    static const uint8_t kCodes[12] = {0,1,2,3,4,5,8,9,10,11,12,13};
    for (size_t i = 0; i < o.NIB.size(); i++) {
        const uint8_t lo = kCodes[r.below(12)], hi = kCodes[r.below(12)];
        o.NIB[i] = lo | (hi << 4);
    }
    for (auto& v : o.BS) v = (uint8_t)(115 + r.below(25));
    for (auto& v : o.A)  v = (uint8_t)(r.below(224) & 0x7e);
    for (auto& v : o.AS) v = 0.001f + 0.01f * (float)r.below(1000) / 1000.f;
    // repack B: one signed i8 per code (x2 doubled)
    if (!want_w8) { o.W8.clear(); }
    else o.W8.resize((size_t)o.N * o.K);
    if (want_w8) for (int n = 0; n < o.N; n++)
        for (int k = 0; k < o.K; k++) {
            const uint8_t b = o.NIB[(size_t)n * (o.K >> 1) + (k >> 1)];
            const uint8_t c = (k & 1) ? (b >> 4) : (b & 0xf);
            if ((c & 7) == 7) { fprintf(stderr, "FATAL: NaN e2m1 code generated\n"); exit(4); }
            o.W8[(size_t)n * o.K + k] = (int8_t)kE2M1X2[c];
        }
    // repack C: 136B groups
    if (!want_s136) { o.S136.clear(); }
    else {
    o.S136.assign((size_t)o.N * (o.K + 8 * K128), 0);
    for (int n = 0; n < o.N; n++) {
        uint8_t* d = o.S136.data() + (size_t)n * (o.K + 8 * K128);
        for (int g128 = 0; g128 < K128; g128++) {
            uint8_t* gr = d + g128 * 136;
            for (int j = 0; j < 4; j++) gr[j] = o.BS[(size_t)n * K32 + g128 * 4 + j];
            memcpy(gr + 8, &o.W8[(size_t)n * o.K + g128 * 128], 128);
        }
    }
    }
    if (!want_nib) { o.NIB.clear(); o.NIB.shrink_to_fit(); }
}

// ---------- the model of the arm's own chain (bit-exact f32 reproduction) ----------

static void model_row(const Op& o, int m, int n, const int8_t* X8, const float* SG,
                      uint16_t* out /*1*/) {
    const int K32 = o.K >> 5, K128 = o.K >> 7, nstep = o.K >> 8;
    // requant the activation row exactly like the prologue
    std::vector<int8_t> x(o.K);
    std::vector<float>  s(K32);
    for (int g = 0; g < K32; g++) {
        float v[32]; float mx = 0.f;
        for (int j = 0; j < 32; j++) { const float e = e4m3_dec(o.A[(size_t)m * o.K + g * 32 + j]); v[j] = e; mx = fmaxf(mx, fabsf(e)); }
        float sg = mx ? mx / 127.0f : 1.0f;
        s[g] = sg;
        for (int j = 0; j < 32; j++) x[g * 32 + j] = (int8_t)rintf(v[j] / sg);
    }
    (void)X8; (void)SG;  // the device chain is verified separately against these
    float acc = 0.f;
    for (int it = 0; it < nstep; it++) {
        float term[8];
        for (int q = 0; q < 8; q++) {
            const int g = it * 8 + q;
            int32_t p = 0;
            for (int lane = 0; lane < 4; lane++) {           // fold the quad by hand
                const int t = q * 4 + lane;
                const size_t base = (size_t)it * 256 + t * 8;
                for (int j = 0; j < 8; j++)
                    p += (int32_t)x[base + j] * (int32_t)o.W8[(size_t)n * o.K + base + j];
            }
            const float sc = bits2f((uint32_t)o.BS[(size_t)n * K32 + g] << 23);
            term[q] = (float)p * s[g] * o.AS[(size_t)m * K128 + (g >> 2)] * sc * 0.5f;
        }
        // the exact butterfly tree: lane off 4,8,16 == quad xor 1,2,4
        // => T = ((t0+t1)+(t2+t3)) + ((t4+t5)+(t6+t7))
        float v[2] = { (term[0] + term[1]) + (term[2] + term[3]),
                       (term[4] + term[5]) + (term[6] + term[7]) };
        acc += v[0] + v[1];
    }
    *out = bf16_rne(acc);
}

// ---------- device launch plumbing --------------------------------------------------

struct Dev {
    int8_t*  dX8; float* dSG; uint16_t* dY; int8_t* dW; uint8_t* dBS; uint8_t* dS136;
    float* dAS; uint8_t* dA;
    std::vector<uint8_t*> rotW; std::vector<uint8_t*> rotS;
};

static void launch(char arm, int mt, int pref, Dev& d, int N, int K, uint8_t* w, hipStream_t st = 0) {
    if (arm == 'A') {
        #define LA(MT_, P_) gemv_A<MT_, P_><<<N, 32, 0, st>>>(w, d.dBS, d.dX8, d.dSG, d.dAS, d.dY, N, K)
        if (mt == 1) { if (pref == 1) { LA(1, 1); } else { LA(1, 2); } }
        else         { if (pref == 1) { LA(4, 1); } else { LA(4, 2); } }
        #undef LA
    } else if (arm == 'B') {
        #define LB(MT_, P_) gemv_B<MT_, P_><<<N, 32, 0, st>>>((const int8_t*)w, d.dBS, d.dX8, d.dSG, d.dAS, d.dY, N, K)
        if (mt == 1) { if (pref == 1) { LB(1, 1); } else { LB(1, 2); } }
        else         { if (pref == 1) { LB(4, 1); } else { LB(4, 2); } }
        #undef LB
    } else {
        #define LC(MT_, P_) gemv_C<MT_, P_><<<N, 32, 0, st>>>(w, d.dX8, d.dSG, d.dAS, d.dY, N, K)
        if (mt == 1) { if (pref == 1) { LC(1, 1); } else { LC(1, 2); } }
        else         { if (pref == 1) { LC(4, 1); } else { LC(4, 2); } }
        #undef LC
    }
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) { fprintf(stderr, "launch %c: %s\n", arm, hipGetErrorName(e)); exit(2); }
}

static void fill_act(Op& o, Dev& d) {
    CK(hipMemcpy(d.dA, o.A.data(), o.A.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(d.dAS, o.AS.data(), o.AS.size() * 4, hipMemcpyHostToDevice));
    prologue_requant<<<((size_t)o.M * (o.K >> 5) + 7) / 8, 256>>>(d.dA, d.dX8, d.dSG, o.M, o.K);
    CK(hipGetLastError()); CK(hipDeviceSynchronize());
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: parity|time|prologue ...\n"); return 1; }
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, 0));
    if (!strstr(prop.name, "7900") && !strstr(prop.gcnArchName, "gfx1100")) {
        fprintf(stderr, "ABORT: device is %s (%s), not gfx1100\n", prop.name, prop.gcnArchName); return 3; }
    printf("DEV %s (%s)\n", prop.name, prop.gcnArchName);

    const std::string mode = argv[1];

    if (mode == "prologue") {                       // prologue <M> <K> <reps>
        const int M = atoi(argv[2]), K = atoi(argv[3]), reps = atoi(argv[4]);
        std::vector<uint8_t> hA((size_t)M * K); Rng r(99);
        for (auto& v : hA) v = (uint8_t)(r.below(224) & 0x7e);
        uint8_t* dA; int8_t* dX8; float* dSG;
        CK(hipMalloc((void**)&dA, hA.size()));
        CK(hipMalloc((void**)&dX8, hA.size()));
        CK(hipMalloc((void**)&dSG, (size_t)M * (K >> 5) * 4));
        CK(hipMemcpy(dA, hA.data(), hA.size(), hipMemcpyHostToDevice));
        prologue_requant<<<((size_t)M * (K >> 5) + 7) / 8, 256>>>(dA, dX8, dSG, M, K);
        CK(hipDeviceSynchronize());
        double t0 = now_s();
        for (int i = 0; i < reps; i++)
            prologue_requant<<<((size_t)M * (K >> 5) + 7) / 8, 256>>>(dA, dX8, dSG, M, K);
        CK(hipDeviceSynchronize());
        double ms = (now_s() - t0) * 1e3 / reps;
        printf("RESULT which=prologue M=%d K=%d per_us=%.1f GBps=%.1f (act bytes=%.1fMB)\n",
               M, K, ms * 1e3, hA.size() / (ms * 1e6), hA.size() / 1e6);
        return 0;
    }

    char arm = 'B';
    int N, K, M, reps = 9, rot = 1, pref = 2;
    if (mode == "parity") { arm = argv[2][0]; N = atoi(argv[3]); K = atoi(argv[4]); M = atoi(argv[5]); }
    else {                                        // time <arm> <M> <K> <N> <reps> <rot> [pref]
        arm = argv[2][0]; M = atoi(argv[3]); K = atoi(argv[4]); N = atoi(argv[5]);
        reps = atoi(argv[6]); rot = atoi(argv[7]); pref = argc > 8 ? atoi(argv[8]) : 2;
    }
    if (K % 256) { fprintf(stderr, "probe gate: K %% 256 == 0\n"); return 1; }
    if (arm != 'A' && arm != 'B' && arm != 'C') { fprintf(stderr, "unknown arm %c\n", arm); return 1; }

    Op o; o.N = N; o.K = K; o.M = M;
    const bool par = (mode == "parity");
    gen_ops(o, 0xC0FFEE ^ (uint64_t)N * 131071 ^ (uint64_t)K,
            /*want_w8*/  par || arm != 'A',
            /*want_s136*/ arm == 'C',
            /*want_nib*/  par || arm == 'A');
    printf("GEN N=%d K=%d M=%d\n", N, K, M);

    Dev d{};
    CK(hipMalloc((void**)&d.dA,  (size_t)M * K));
    CK(hipMalloc((void**)&d.dAS, (size_t)M * (K >> 7) * 4));
    CK(hipMalloc((void**)&d.dX8, (size_t)M * K));
    CK(hipMalloc((void**)&d.dSG, (size_t)M * (K >> 5) * 4));
    CK(hipMalloc((void**)&d.dY,  (size_t)M * N * 2));
    if (arm == 'A' || arm == 'B') {
        CK(hipMalloc((void**)&d.dBS, o.BS.size()));
        CK(hipMemcpy(d.dBS, o.BS.data(), o.BS.size(), hipMemcpyHostToDevice));
        const uint8_t* srcp; size_t srcn;
        if (arm == 'A') { srcp = o.NIB.data(); srcn = o.NIB.size(); }
        else            { srcp = (const uint8_t*)o.W8.data(); srcn = o.W8.size(); }
        for (int i = 0; i < rot; i++) { uint8_t* p = nullptr; CK(hipMalloc((void**)&p, srcn));
            CK(hipMemcpy(p, srcp, srcn, hipMemcpyHostToDevice)); d.rotW.push_back(p); }
    } else {
        for (int i = 0; i < rot; i++) { uint8_t* p; CK(hipMalloc((void**)&p, o.S136.size()));
            CK(hipMemcpy(p, o.S136.data(), o.S136.size(), hipMemcpyHostToDevice)); d.rotS.push_back(p); }
    }
    fill_act(o, d);

    if (mode == "parity") {
        uint8_t* w = arm == 'C' ? d.rotS[0] : d.rotW[0];
        int rc = 0;
        for (int pref_p : {1, 2}) {
            CK(hipMemset(d.dY, 0xFF, (size_t)M * N * 2));
            launch(arm, M, pref_p, d, N, K, w); CK(hipDeviceSynchronize());
            std::vector<uint16_t> hY((size_t)M * N);
            CK(hipMemcpy(hY.data(), d.dY, hY.size() * 2, hipMemcpyDeviceToHost));
            long poison = 0, mism = 0; double s2 = 0, ss = 0;
            for (int m = 0; m < M; m++)
                for (int n = 0; n < N; n++) {
                    if (hY[(size_t)m * N + n] == 0xFFFF) poison++;
                    uint16_t want; model_row(o, m, n, nullptr, nullptr, &want);
                    const double a = (double)bits2f((uint32_t)hY[(size_t)m * N + n] << 16);
                    const double b = (double)bits2f((uint32_t)want << 16);
                    if (hY[(size_t)m * N + n] != want) { mism++;
                        if (mism <= 3) printf("  MISMATCH m=%d n=%d gpu=%.6g model=%.6g\n", m, n, a, b); }
                    s2 += (a - b) * (a - b); ss += b * b;
                }
            printf("PARITY arm=%c N=%d K=%d M=%d pref=%d mismatches=%ld/%ld rel_l2=%.3e poison_untouched=%ld %s\n",
                   arm, N, K, M, pref_p, mism, (long)(M * N), sqrt(s2 / (ss + 1e-30)), poison,
                   (mism == 0 && poison == 0) ? "OK" : "FAIL");
            if (mism || poison) rc = 4;
        }
        return rc;
    }

    // timing — methodology mirrored EXACTLY from bench_gemv_prof: one warm launch
    // (plane 0), then reps launches with rotation and a single final sync, ms/reps.
    uint8_t** plane = arm == 'C' ? d.rotS.data() : d.rotW.data();
    launch(arm, M, pref, d, N, K, plane[0]); CK(hipDeviceSynchronize());
    double t0 = now_s();
    for (int i = 0; i < reps; i++) launch(arm, M, pref, d, N, K, plane[i % rot]);
    CK(hipDeviceSynchronize());
    const double best = (now_s() - t0) / reps;
    const double wbytes = arm == 'A' ? (double)N * (K >> 1) + (double)N * (K >> 5)
                         : arm == 'B' ? (double)N * K + (double)N * (K >> 5)
                                      : (double)N * (K + 8 * (K >> 7));
    const double fp4b   = (double)N * (K >> 1);          // bench_gemv_prof's convention
    printf("RESULT which=%c pref=%d M=%d K=%d N=%d rot=%d reps=%d per_ms=%.3f "
           "GBps=%.1f GBps_fp4eq=%.1f (plane=%.1fMB)\n",
           arm, pref, M, K, N, rot, reps, best * 1e3,
           wbytes / best / 1e9, fp4b / best / 1e9, wbytes / 1048576.0);
    return 0;
}
