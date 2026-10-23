/* probe_mlp.hip — standalone memory-level-parallelism discriminator for the
 * decode-GEMV plateau (model-band shapes 110-235 GB/s vs ~800 GB/s pure-walk).
 * NO plugin, NO ABI: pure hipcc single TU.
 *
 * Reproduces the GEMV's walk geometry (wave32 per weight row, 256-thr blocks,
 * 8 warps) with the math peeled off in tiers, so one run says WHICH term caps
 * the plateau:
 *   T0: weight walk only              -> pure load-shape ceiling
 *   T1: + LDS LUT lookups (16/byte)   -> LDS term
 *   T2: T1 + fold tree                -> reduction term
 *   T4: weight walk + act stream      -> load-mix, no math
 *   T3: T2 + act stream               -> full load mix, fake math
 *   T50: the REAL MT=1 consume_span verbatim (LDS LUT + fp8 decode + FMAs +
 *        pair/warp fold + AS scalar)          -> should ~= the shipped GEMV
 *   T51: T50 minus the AS global load
 *   T52: T50 with LDS LUT replaced by __constant__ LUT
 * Load-width arms LDX in {8,16,32} B/lane, split arms S (wave-count xS),
 * RP in {1,2} (two row streams half-plane apart, each row read exactly once).
 *
 * Plane discipline: serve regime has no L2 reuse => rotate distinct B planes
 * until footprint >= 96 MB (> 64 MB Infinity Cache). Rate = weight bytes / min
 * launch wall (event-timed).
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorName(e_)); exit(2); } } while (0)

__constant__ float kLUT[16];
static const float kLUTH[16] = {0,0.5f,1,1.5f,2,3,4,6,-0,-0.5f,-1,-1.5f,-2,-3,-4,-6};

/* verbatim from r3d_gemm_mxfp4.hip (fp8e4m3_gemv): the real act-byte decoder */
__device__ __forceinline__ float fp8e4m3_gemv_p(uint8_t b) {
    const uint32_t s = (uint32_t)(b >> 7) << 31;
    const uint32_t e = (uint32_t)(b >> 3) & 0xfu, m = (uint32_t)b & 0x7u;
    uint32_t u;
    if (e == 0u) {
        if (m == 0u) return __uint_as_float(s);
        const float f = (float)m / 512.0f;
        u = s | (__float_as_uint(f) & 0x7fffffffu);
    } else if (e == 0xfu && m == 0x7u) {
        u = s | 0x7fc00000u;
    } else {
        u = s | ((e - 7u + 127u) << 23) | (m << 20);
    }
    return __uint_as_float(u);
}

/* TIER: 0 walk | 1 +LDS LUT | 2 +fold | 3 T2+act | 4 walk+act | 50 real span
 * body | 51 no-AS | 52 c-cache LUT.  LDX weight bytes/lane (50/51/52 pin 8). */
template <int LDX, int TIER, int S, int RP>
__global__ __launch_bounds__(256) void walk(const uint8_t* __restrict__ B,
                                            const uint8_t* __restrict__ A,
                                            const float* __restrict__ AS,
                                            float* __restrict__ out,
                                            int N, int K, int64_t b_pitch)
{
    __shared__ float L16[16];
    if (threadIdx.x < 16) L16[threadIdx.x] = kLUTH[threadIdx.x];

    const int w    = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nb   = gridDim.x / S;
    const int s    = blockIdx.x / nb;
    const int task = (blockIdx.x % nb) * 8 + w;

    const int64_t halfK = K >> 1;
    const int64_t lo    = halfK * s / S;
    const int64_t hi    = halfK * (s + 1) / S;
    const int     niter = (int)(((hi - lo) + (int64_t)LDX * 32 - 1) / ((int64_t)LDX * 32));

    float acc[RP];
#pragma unroll
    for (int r = 0; r < RP; ++r) acc[r] = 0.0f;

    for (int it = 0; it < niter; ++it) {
        const int64_t off = lo + (int64_t)it * (LDX * 32) + (int64_t)lane * LDX;
        const bool act = off + LDX <= hi;
#pragma unroll
        for (int r = 0; r < RP; ++r) {
            const int row = task + r * (N / RP);   /* streams half-plane apart */
            if (row >= N) break;
            const uint8_t* p = B + (int64_t)row * b_pitch + off;
            uint32_t v0 = 0, v1 = 0, v2 = 0, v3 = 0;
            if (act) {
                if (LDX >= 8)  { v0 = *(const uint32_t*)(p); v1 = *(const uint32_t*)(p + 4); }
                if (LDX >= 16) { v2 = *(const uint32_t*)(p + 8);  v3 = *(const uint32_t*)(p + 12); }
                if (LDX >= 32) { v0 ^= *(const uint32_t*)(p + 16); v1 ^= *(const uint32_t*)(p + 20);
                                 v2 ^= *(const uint32_t*)(p + 24); v3 ^= *(const uint32_t*)(p + 28); }
            }
            if constexpr (TIER >= 50) {
                /* ---- real MT=1 consume_span, off == the lane's weight byte ---- */
                const uint2 b8 = act ? *(const uint2*)(B + (int64_t)row * b_pitch + off)
                                     : make_uint2(0u, 0u);
                uint4 av = make_uint4(0u, 0u, 0u, 0u);
                if (act && off * 2 + 16 <= hi * 2) av = *(const uint4*)(A + off * 2);
                float sv;
                const uint8_t* ax = (const uint8_t*)&av;
                if constexpr (TIER == 53 || TIER == 54) {
                    /* chain split: 2 (or 4) independent FMA chains, interleaved
                     * by the scheduler; s = sum of partials (re-associated) */
                    constexpr int NC = (TIER == 53) ? 2 : 4;
                    float sc_[NC];
#pragma unroll
                    for (int q = 0; q < NC; ++q) sc_[q] = 0.0f;
#pragma unroll
                    for (int t = 0; t < 16; ++t)
                        sc_[t % NC] = __fmaf_rn(fp8e4m3_gemv_p(ax[t]), L16[((t < 8 ? b8.x : b8.y) >> (4 * (t & 7))) & 15u], sc_[t % NC]);
                    sv = sc_[0];
#pragma unroll
                    for (int q = 1; q < NC; ++q) sv += sc_[q];
                } else {
                    sv = 0.0f;
#pragma unroll
                    for (int t = 0; t < 16; ++t) {
                        const uint32_t nib = (t < 8 ? b8.x : b8.y) >> (4 * (t & 7));
                        const float wgt = (TIER == 52) ? kLUT[nib & 15u] : L16[nib & 15u];
                        sv = __fmaf_rn(fp8e4m3_gemv_p(ax[t]), wgt, sv);
                    }
                }
                sv += __shfl_xor(sv, 1);
                const int g = lane >> 1;
                float c = (TIER == 51) ? sv * 1.0001f
                                       : (sv * AS[(g >> 2) & 15]) * (float)(lane + 128);
#pragma unroll
                for (int o2 = 2; o2 < 32; o2 <<= 1) c += __shfl_xor(c, o2);
                acc[r] += c;
            } else if constexpr (TIER == 4) {
                acc[r] += (float)(v0 ^ v1 ^ v2 ^ v3);
                const int64_t aoff = off * 2;
                if (aoff + LDX * 2 <= hi * 2) {
                    const uint4 av = *(const uint4*)(A + aoff);
                    acc[r] += (float)(av.x ^ av.y ^ av.z ^ av.w);
                }
            } else if constexpr (TIER == 0) {
                acc[r] += (float)(v0 ^ v1 ^ v2 ^ v3);
            } else {
                float s0 = 0;
                const uint32_t words[4] = {v0, v1, v2, v3};
#pragma unroll
                for (int t = 0; t < LDX / 4; ++t)
#pragma unroll
                    for (int h = 0; h < 8; ++h) {
                        s0 += L16[(words[t] >> (4 * h)) & 15u];
                        if constexpr (TIER == 2) s0 *= L16[words[t] & 15u];
                    }
                if constexpr (TIER < 2) {
                    acc[r] += s0;
                } else {
                    float c = s0 + __shfl_xor(s0, 1);
#pragma unroll
                    for (int o = 2; o < 32; o <<= 1) c += __shfl_xor(c, o);
                    acc[r] += c;
                }
                if constexpr (TIER >= 3) {
                    const int64_t aoff = off * 2;
                    if (aoff + LDX * 2 <= hi * 2) {
                        const uint4 av = *(const uint4*)(A + aoff);
                        acc[r] += (float)(av.x ^ av.y ^ av.z ^ av.w);
                    }
                }
            }
        }
    }
    float t = acc[0];
#pragma unroll
    for (int r = 1; r < RP; ++r) t += acc[r];
    if (lane == 0) out[blockIdx.x * 8 + w] = t;
}

struct Shape { int N, K; const char* tag; };

template <int LDX, int TIER, int S, int RP>
double run_one(const uint8_t* const* dB, const uint8_t* dA, const float* dAS,
               float* dO, int N, int K, int rot, int reps)
{
    const int nb = (N + RP * 8 - 1) / (RP * 8);
    dim3 grid(nb * S), block(256);
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    float best = 1e30f;
    for (int i = -3; i < reps; ++i) {
        const uint8_t* B = dB[i > 0 ? i % rot : 0];
        CK(hipDeviceSynchronize());
        CK(hipEventRecord(e0));
        walk<LDX, TIER, S, RP><<<grid, block>>>(B, dA, dAS, dO, N, K, K >> 1);
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        float ms; CK(hipEventElapsedTime(&ms, e0, e1));
        best = std::min(best, ms);
    }
    CK(hipEventDestroy(e0)); CK(hipEventDestroy(e1));
    return (double)N * (K / 2.0) / (best * 1e-3) / 1e9;
}

int main(int argc, char** argv)
{
    static const Shape shapes[] = {
        { 4096,  4096, "qkv-tight 4096x4096"},
        {12288,  4096, "qkv-wide  12288x4096"},
        { 4096, 12288, "ffn-tall   4096x12288"},
        {24576,  4096, "ffn-wide  24576x4096"},
        {16384, 16384, "giant     16384x16384"},
    };
    float hLUT[16]; for (int i = 0; i < 16; ++i) hLUT[i] = kLUTH[i];
    CK(hipMemcpyToSymbol(kLUT, hLUT, sizeof hLUT));
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, 0));
    printf("DEV %s CUs=%d l2=%.0fMB\n", prop.name, prop.multiProcessorCount,
           prop.l2CacheSize / 1048576.0);

    const int reps = argc > 1 ? atoi(argv[1]) : 20;
    printf("%-26s %-28s %8s\n", "shape", "arm (ldx/tier/S/RP)", "GBps");
    for (const Shape& shp : shapes) {
        const size_t plane = (size_t)shp.N * (shp.K / 2);
        int rot = (int)((96u * 1048576 + plane - 1) / plane);
        if (rot < 2) rot = 2; if (rot > 4) rot = 4;
        std::vector<uint8_t*> dB(rot);
        for (int r = 0; r < rot; ++r) CK(hipMalloc((void**)&dB[r], plane));
        std::vector<uint8_t> h(plane);
        for (size_t i = 0; i < plane; ++i) h[i] = (uint8_t)rand();
        for (int r = 0; r < rot; ++r) CK(hipMemcpy(dB[r], h.data(), plane, hipMemcpyHostToDevice));
        uint8_t* dA; CK(hipMalloc((void**)&dA, (size_t)shp.K * 4));
        CK(hipMemset(dA, 0x30, (size_t)shp.K * 4));
        float* dAS; CK(hipMalloc((void**)&dAS, 4096));
        CK(hipMemset(dAS, 0x3c, 4096));
        float* dO; CK(hipMalloc((void**)&dO, 4 * 65536));

#define ARM(LDX, TIER, S, RP) printf("%-26s %-28s %8.1f\n", shp.tag, \
        #LDX "/" #TIER "/" #S "/" #RP, run_one<LDX, TIER, S, RP>(dB.data(), dA, dAS, dO, shp.N, shp.K, rot, reps));
        ARM(8, 0, 1, 1)
        ARM(8, 1, 1, 1)
        ARM(8, 3, 1, 1)
        ARM(8, 50, 1, 1)  /* the real span body: expect ~= shipped GEMV */
        ARM(8, 51, 1, 1)
        ARM(8, 52, 1, 1)
        ARM(8, 53, 1, 1)  /* chain split x2: the latency discriminator */
        ARM(8, 54, 1, 1)  /* chain split x4 */
        ARM(8, 53, 1, 2)  /* chain split + row pair */
        ARM(8, 4, 1, 1)
        ARM(16, 4, 1, 1)
        ARM(8, 50, 2, 1)  /* real body + splits */
        ARM(8, 50, 1, 2)  /* real body + row pairs */
#undef ARM
        for (int r = 0; r < rot; ++r) CK(hipFree(dB[r]));
        CK(hipFree(dA)); CK(hipFree(dAS)); CK(hipFree(dO));
    }
    return 0;
}
