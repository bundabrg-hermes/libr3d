// probe_wmma.cpp -- Q21 pre-adjudication + Q11 lane-map unblock (REVIEW_QUEUE).
//
// Two stages, one binary (window 17):
//   maps   -- pin the WMMA fragment lane map on gfx1100 for this ROCm's rocwmma on
//             wave32: which (row, col) each lane's accumulator registers hold, and
//             which (row, k) the row-major A fragment's registers hold. The Q11
//             epics (fragment-side softmax, VGPR group flush) were gated on exactly
//             this datum; rocWMMA exposes no coordinate fragment, so we manufacture
//             one: A[i][k] = i*16 + k (exact in bf16 for i,k < 16), B = identity,
//             gemm -> C[i][j] = i*16 + j, then dump every lane's fragment registers.
//   rates  -- peak sustained throughput of the compute engines that compete on the
//             decode-GEMM space: v_wmma bf16 / f16 / i8 / u8 (16x16x16) against the
//             shipped v_dot4 chain (v_wmma_i32_iu8 spelling via sudot4), same block
//             geometry, full device. Q21's skinny-GEMM build/no-build hinges on this:
//             the A/B benches showed the w2a8 row is COMPUTE-bound in the verify band
//             (M=14 pays 4x M=1's time on identical weight traffic), so the engine
//             rate is the ceiling; the i8-vs-f16 sub-question is the triage doc's
//             open "iu8 no faster than f16?" claim, unmeasured here until now.
//
// Build (radiance-dev container, the library's flags: wave32 + gfx1100):
//   hipcc -O2 -std=c++17 --offload-arch=gfx1100 -mno-wavefrontsize64 -w \
//         tools/probe_wmma.cpp -o /tmp/pwmma
// Run (card exclusive: stop spec9g first):
//   ROCR_VISIBLE_DEVICES=1 /tmp/pwmma maps
//   ROCR_VISIBLE_DEVICES=1 /tmp/pwmma rates [iters]
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace roCW = rocwmma;

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP %s: %s\n", #x, hipGetErrorName(e_)); return 1; } } while (0)

/* ------------------------------------------------------------------ maps ---- */
/* One wave32. Dump raw fragment element registers per lane; host decodes.     */
__global__ void map_kernel(float* acc_out, float* a_out) {
    // accumulator map via the identity-gemm trick
    __shared__ __align__(16) uint16_t Ab[16 * 16], Ib[16 * 16];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        const int r = i >> 4, c = i & 15;
        // bf16 bits of (float)(r*16+c): f32 top 16 bits -- EXACT for 0..255
        float v = (float)(r * 16 + c);
        uint32_t u; memcpy(&u, &v, 4);
        Ab[i] = (uint16_t)(u >> 16);
        Ib[i] = (uint16_t)(((r == c) ? 1.0f : 0.0f) == 1.0f ? 0x3F80u : 0x0000u);
    }
    __syncthreads();
    roCW::fragment<roCW::matrix_a, 16, 16, 16, roCW::bfloat16_t, roCW::row_major> fa;
    roCW::fragment<roCW::matrix_b, 16, 16, 16, roCW::bfloat16_t, roCW::col_major> fb;
    roCW::fragment<roCW::accumulator, 16, 16, 16, float> fc;
    // A-fragment map: load the same buffer row-major, dump its registers too
    roCW::load_matrix_sync(fa, (const roCW::bfloat16_t*)Ab, 16);
    roCW::load_matrix_sync(fb, (const roCW::bfloat16_t*)Ib, 16);
    roCW::fill_fragment(fc, 0.0f);
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int k = 0; k < fa.num_elements; ++k)
        a_out[lane * 16 + k] = float(fa.x[k]);
    roCW::mma_sync(fc, fa, fb, fc);            // C[i][j] = A[i][j] = i*16+j
#pragma unroll
    for (int k = 0; k < fc.num_elements; ++k)
        acc_out[lane * 16 + k] = fc.x[k];
}

/* 8-bit variant of the map stage: the skinny arm depends on the i32-accumulator
   map and the u8/i8 fragment maps, which may differ from the bf16/f32 maps. */
__global__ void imap_kernel(float* acc_out, float* a_out, float* b_out) {
    __shared__ __align__(16) uint8_t Ab[16 * 16], Ib[16 * 16];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        const int r = i >> 4, c = i & 15;
        Ab[i] = (uint8_t)(r * 16 + c);                    // 0..255 exact in u8
        Ib[i] = (uint8_t)((r == c) ? 1 : 0);
    }
    __syncthreads();
    // int8 x int8 ONLY: this rocwmma's u8 x u8 mma_sync lowers to NOTHING on the
    // RDNA3 backend (imap with u8 gave an all-zero accumulator; the rate stage's
    // "infinite" u8 number was the same no-op). Loads are real for both; mma i8 only.
    roCW::fragment<roCW::matrix_a, 16, 16, 16, int8_t, roCW::row_major> fa;
    roCW::fragment<roCW::matrix_b, 16, 16, 16, int8_t, roCW::col_major> fb;
    roCW::fragment<roCW::accumulator, 16, 16, 16, int32_t> fc;
    roCW::load_matrix_sync(fa, (const int8_t*)Ab, 16);
    roCW::load_matrix_sync(fb, (const int8_t*)Ib, 16);
    roCW::fill_fragment(fc, (int32_t)0);
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int k = 0; k < fa.num_elements; ++k) a_out[lane * 16 + k] = (float)fa.x[k];
#pragma unroll
    for (int k = 0; k < fb.num_elements; ++k)
        b_out[lane * 16 + k] = (float)(int)fb.x[k];       // 0/1 identity
    roCW::mma_sync(fc, fa, fb, fc);                       // C[i][j] = i*16 + j
#pragma unroll
    for (int k = 0; k < fc.num_elements; ++k) acc_out[lane * 16 + k] = (float)fc.x[k];
}

int stage_imaps() {
    float *acc, *afr, *bfr;
    CK(hipMalloc(&acc, 32 * 16 * sizeof(float)));
    CK(hipMalloc(&afr, 32 * 16 * sizeof(float)));
    CK(hipMalloc(&bfr, 32 * 16 * sizeof(float)));
    imap_kernel<<<1, 32>>>(acc, afr, bfr);
    CK(hipDeviceSynchronize());
    float h[32 * 16];
    CK(hipMemcpy(h, acc, sizeof(h), hipMemcpyDeviceToHost));
    printf("== ACCUMULATOR i32 lane map (i8 x i8 gemm, C[i][j]=(i*16+j) wrapped i8) ==\n");
    for (int l = 0; l < 32; ++l) {
        printf("lane %2d:", l);
        for (int k = 0; k < 16; ++k) {
            int v = (int)h[l * 16 + k];
            if (v < 0) v += 256;
            if (v >= 0 && v <= 255) printf(" r%d=(%d,%d)", k, v / 16, v % 16);
        }
        printf("\n");
    }
    CK(hipMemcpy(h, afr, sizeof(h), hipMemcpyDeviceToHost));
    printf("== MATRIX_A i8 row_major map (A[i][k]=i*16+k wrapped) ==\n");
    for (int l = 0; l < 32; ++l) {
        printf("lane %2d:", l);
        for (int k = 0; k < 16; ++k) {
            int v = (int)h[l * 16 + k];
            if (v > 127) v -= 256;                        // x[] prints signed i8 domain
            if (v < 0) v += 256;
            if (v >= 0 && v <= 255) printf(" r%d=(%d,%d)", k, v / 16, v % 16);
        }
        printf("\n");
    }
    CK(hipMemcpy(h, bfr, sizeof(h), hipMemcpyDeviceToHost));
    printf("== MATRIX_B u8 col_major map: identity carries no coords (values 0/1);\n"
           "   instead dump which (k,j) produce k-index: skipped, A+B maps + acc close it.\n");
    hipFree(acc); hipFree(afr); hipFree(bfr);
    return 0;
}

/* B-contract probe: does matrix_b col_major + ldm=136 read (k,j) at ptr[j*ldm + k]
   (what the skinny kernel assumes: token-major staging), or the reverse? Stages B so
   that the staged slot i of "token" j carries value j*16+i, pairs it with an identity
   A, and checks C[row][col] == col*16 + row (CONFIRMED) or row*16 + col (INVERTED). */
__global__ void bmap_kernel(float* acc_out) {
    __shared__ __align__(16) int8_t Aid[16 * 16], Bs[16 * 136];
    for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x)
        Aid[i] = (int8_t)(((i >> 4) == (i & 15)) ? 1 : 0);
    for (int i = threadIdx.x; i < 16 * 136; i += blockDim.x) {
        const int j = i / 136, k = i % 136;                 // token-major staging
        Bs[i] = (k < 16) ? (int8_t)(j * 16 + k) : (int8_t)0;
    }
    __syncthreads();
    roCW::fragment<roCW::matrix_a, 16, 16, 16, int8_t, roCW::row_major> fa;
    roCW::fragment<roCW::matrix_b, 16, 16, 16, int8_t, roCW::col_major> fb;
    roCW::fragment<roCW::accumulator, 16, 16, 16, int32_t> fc;
    roCW::load_matrix_sync(fa, (const int8_t*)Aid, 16);
    roCW::load_matrix_sync(fb, (const int8_t*)Bs, 136);
    roCW::fill_fragment(fc, (int32_t)0);
    roCW::mma_sync(fc, fa, fb, fc);
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int k = 0; k < fc.num_elements; ++k)
        acc_out[lane * 16 + k] = (float)fc.x[k];
}

int stage_bmap() {
    float* acc;
    CK(hipMalloc(&acc, 32 * 16 * sizeof(float)));
    bmap_kernel<<<1, 32>>>(acc);
    CK(hipDeviceSynchronize());
    float h[32 * 16];
    CK(hipMemcpy(h, acc, sizeof(h), hipMemcpyDeviceToHost));
    long confirmed = 0, inverted = 0, other = 0;
    for (int l = 0; l < 32; ++l)
        for (int k = 0; k < 8; ++k) {
            int v = (int)h[l * 16 + k]; if (v < 0) v += 256;
            const int row = 2 * k + (l >> 4), col = l & 15;
            confirmed += (v == col * 16 + row);
            inverted  += (v == row * 16 + col);
            other     += (v != col * 16 + row && v != row * 16 + col);
        }
    printf("B col_major contract: CONFIRMED(ptr[j*ldm+k], token-major staging) %ld/256"
           "  INVERTED(ptr[k*ldm+j]) %ld/256  OTHER %ld/256\n", confirmed, inverted, other);
    hipFree(acc);
    return 0;
}

int stage_maps() {
    float *acc, *afr;
    CK(hipMalloc(&acc, 32 * 16 * sizeof(float)));
    CK(hipMalloc(&afr, 32 * 16 * sizeof(float)));
    map_kernel<<<1, 32>>>(acc, afr);
    CK(hipDeviceSynchronize());
    float h[32 * 16];
    CK(hipMemcpy(h, acc, sizeof(h), hipMemcpyDeviceToHost));
    printf("== ACCUMULATOR lane map (lane: reg->(row,col); -1=pad) ==\n");
    for (int l = 0; l < 32; ++l) {
        printf("lane %2d:", l);
        for (int k = 0; k < 16; ++k) {
            const int v = (int)h[l * 16 + k];
            if (v >= 0 && v <= 255) printf(" r%d=(%d,%d)", k, v / 16, v % 16);
        }
        printf("\n");
    }
    CK(hipMemcpy(h, afr, sizeof(h), hipMemcpyDeviceToHost));
    printf("== MATRIX_A row-major fragment lane map (A[i][k]=i*16+k) ==\n");
    for (int l = 0; l < 32; ++l) {
        printf("lane %2d:", l);
        for (int k = 0; k < 16; ++k) {
            const int v = (int)h[l * 16 + k];
            if (v >= 0 && v <= 255) printf(" r%d=(%d,%d)", k, v / 16, v % 16);
        }
        printf("\n");
    }
    hipFree(acc); hipFree(afr);
    return 0;
}

/* ------------------------------------------------------------------ rates --- */
template <typename T, typename ACC>
__global__ void wmma_rate_kernel(float* out, int iters, int seed) {
    // 8 waves, 4 independent accumulator chains per wave (pipeline, not latency)
    roCW::fragment<roCW::matrix_a, 16, 16, 16, T, roCW::row_major> fa;
    roCW::fragment<roCW::matrix_b, 16, 16, 16, T, roCW::col_major> fb;
    roCW::fragment<roCW::accumulator, 16, 16, 16, ACC> a0, a1, a2, a3;
    roCW::fill_fragment(fa, T(1)); roCW::fill_fragment(fb, T(1));
    roCW::fill_fragment(a0, ACC(0)); roCW::fill_fragment(a1, ACC(0));
    roCW::fill_fragment(a2, ACC(0)); roCW::fill_fragment(a3, ACC(0));
    for (int i = 0; i < iters; ++i) {
        roCW::mma_sync(a0, fa, fb, a0);
        roCW::mma_sync(a1, fa, fb, a1);
        roCW::mma_sync(a2, fa, fb, a2);
        roCW::mma_sync(a3, fa, fb, a3);
    }
    if (seed) {
        float s = 0;
#pragma unroll
        for (int k = 0; k < a0.num_elements; ++k)
            s += float(a0.x[k]) + float(a1.x[k]) + float(a2.x[k]) + float(a3.x[k]);
        out[blockIdx.x * blockDim.x + threadIdx.x] = s;
    }
}

// the shipped decode engine: v_dot4 chains (signed A x unsigned B like sudot4 in
// r3d_gemm_w2a8.hip), 4 independent accumulator chains per lane.
__global__ void dot4_rate_kernel(float* out, int iters, int seed) {
    int s0 = seed, s1 = ~seed, s2 = seed * 3, s3 = seed ^ 0x5a;
    const int a = 0x017f7f01, b = 0x03020201;
    for (int i = 0; i < iters; ++i) {
        s0 = __builtin_amdgcn_sudot4(1, a, 0, b, s0, 0);
        s1 = __builtin_amdgcn_sudot4(1, a, 0, b, s1, 0);
        s2 = __builtin_amdgcn_sudot4(1, a, 0, b, s2, 0);
        s3 = __builtin_amdgcn_sudot4(1, a, 0, b, s3, 0);
    }
    if (seed) out[blockIdx.x * blockDim.x + threadIdx.x] =
        (float)(s0 + s1 + s2 + s3);
}

int g_blocks = 1, g_threads = 256;
long g_live = -1;
hipEvent_t t0, t1;

float* g_out = nullptr;
template <typename F>
float timeit(F launch, double macs_per_launch, int reps) {
    CK(hipMemset(g_out, 0, sizeof(float) * 4096 * 256));  // dead stages must not
    launch();                                    //   inherit the PREVIOUS stage's
    CK(hipDeviceSynchronize());                  //   nonzero out[] (false liveness)
    hipEventRecord(t0);
    for (int r = 0; r < reps; ++r) launch();
    hipEventRecord(t1);
    CK(hipEventSynchronize(t1));
    float ms; CK(hipEventElapsedTime(&ms, t0, t1));
    // LIVENESS (card protocol): a rate from a dead kernel is a lie. Every live
    // wave wrote out[] with a nonzero accumulator sum; count them.
    static float* h = nullptr;
    if (!h) h = (float*)malloc(sizeof(float) * 4096 * 256);
    CK(hipMemcpy(h, g_out, sizeof(float) * 4096 * 256, hipMemcpyDeviceToHost));
    long nz = 0;
    for (size_t i = 0; i < (size_t)g_blocks * g_threads && i < 4096 * 256; ++i)
        nz += (h[i] != 0.0f);
    g_live = nz;
    return (float)(macs_per_launch * reps / (ms * 1e-3) / 1e9);   // GMAC/s
}

int stage_rates(int iters) {
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0));
    printf("device=%s CUs=%d\n", p.name, p.multiProcessorCount);
    float* out; CK(hipMalloc(&out, (size_t)4096 * 256 * sizeof(float)));
    CK(hipMemset(out, 0, (size_t)4096 * 256 * sizeof(float)));
    const int blocks = p.multiProcessorCount * 4, thr = 256;   // 8 waves/blk
    g_blocks = blocks; g_threads = thr; g_out = out;
    const double waves = (double)blocks * thr / 32.0;
    const double macs_wmma = waves * 4.0 * iters * 4096.0;     // 4 gemm/wave/iter
    const double macs_dot4 = waves * 4.0 * iters * 4.0 * 32.0; // 4 dot4*4MAC/lane
    int reps = 3;
    float r;
    r = timeit([&]{ (wmma_rate_kernel<roCW::bfloat16_t, float><<<blocks,thr>>>(out, iters, 1)); }, macs_wmma, reps);
    printf("wmma bf16 16x16x16 : %8.1f GMAC/s (%6.1f GFLOP/s x2) live=%ld/%d\n", r, 2*r, g_live, blocks*thr);
    r = timeit([&]{ (wmma_rate_kernel<__half, float><<<blocks,thr>>>(out, iters, 1)); }, macs_wmma, reps);
    printf("wmma f16  16x16x16 : %8.1f GMAC/s live=%ld/%d\n", r, g_live, blocks*thr);
    r = timeit([&]{ (dot4_rate_kernel<<<blocks,thr>>>(out, iters, 1)); }, macs_dot4, reps);
    printf("dot4 (sudot4 chain): %8.1f GMAC/s live=%ld/%d\n", r, g_live, blocks*thr);
    {   // i8 engines: only if this rocwmma carries the types (triage open question)
        r = timeit([&]{ (wmma_rate_kernel<int8_t, int32_t><<<blocks,thr>>>(out, iters, 1)); }, macs_wmma, reps);
        printf("wmma i8*s8 i32 acc : %8.1f GMAC/s live=%ld/%d\n", r, g_live, blocks*thr);
        r = timeit([&]{ (wmma_rate_kernel<uint8_t, int32_t><<<blocks,thr>>>(out, iters, 1)); }, macs_wmma, reps);
        printf("wmma u8*u8 i32 acc : %8.1f GMAC/s live=%ld/%d\n", r, g_live, blocks*thr);
    }
    CK(hipGetLastError());
    hipFree(out);
    return 0;
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "all";
    const int iters = argc > 2 ? atoi(argv[2]) : 20000;
    CK(hipInit(0));
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0));
    printf("device=%s wavefront=%d\n", p.name, p.warpSize);
    CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));
    if (strcmp(mode, "rates") != 0) stage_maps();
    if (strcmp(mode, "maps") != 0 && strcmp(mode, "rates") != 0) { stage_imaps(); stage_bmap(); }
    if (strcmp(mode, "maps") != 0 && strcmp(mode, "imaps") != 0) stage_rates(iters);
    return 0;
}
