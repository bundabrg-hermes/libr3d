// probe_l401b.hip -- isolate hipErrorIllegalState(401) on the FA kernel's exact shape:
// static LDS ~41 KB, 128 threads, launch_bounds, rocwmma. Axes tested by BUILD, not code:
//   A: single-arch (gfx1100), linked binary
//   B: multi-arch (gfx1100 + gfx1101 + gfx11-generic), linked binary
//   C: multi-arch -fPIC -shared .so, launched via dlopen from a host main
#include <hip/hip_runtime.h>
#include <cstdio>

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
#include <rocwmma/rocwmma.hpp>
namespace rw = rocwmma;
extern "C" __global__ __launch_bounds__(128) void k41(rw::bfloat16_t* o) {
    __shared__ unsigned char raw[41008];                        /* the FA carve, byte-exact */
    rw::bfloat16_t* s = reinterpret_cast<rw::bfloat16_t*>(raw);
    rw::fragment<rw::accumulator, 16, 16, 16, float> acc[8];
    rw::fragment<rw::matrix_a, 16, 16, 16, rw::bfloat16_t, rw::row_major> fa;
    rw::fragment<rw::matrix_b, 16, 16, 16, rw::bfloat16_t, rw::col_major> fb[8];
    for (int i = 0; i < 8; ++i) { rw::fill_fragment(acc[i], 0.0f); rw::fill_fragment(fb[i], rw::bfloat16_t(1.0f)); }
    rw::fill_fragment(fa, rw::bfloat16_t(1.0f));
    #pragma unroll
    for (int k = 0; k < 8; ++k) rw::mma_sync(acc[k], fa, fb[k], acc[k]);
    float* fs = reinterpret_cast<float*>(raw + 32768);          /* f32 staging inside carve */
    for (int i = 0; i < 8; ++i)
        rw::store_matrix_sync(fs + threadIdx.x * 8 + i, acc[i], 64, rw::mem_row_major);
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0;
        for (int i = 0; i < 64; ++i) t += fs[i] + float(s[i]);
        o[0] = rw::bfloat16_t(t);
    }
}
#else
namespace rw { typedef unsigned short bfloat16_t; }
extern "C" __global__ __launch_bounds__(128) void k41(rw::bfloat16_t*) {}
#endif

extern "C" int run_probe() {
    int dev = 0; hipDeviceProp_t p{};
    hipGetDevice(&dev); hipGetDeviceProperties(&p, dev);
    rw::bfloat16_t* d; hipMalloc(&d, 2);
    hipFuncAttributes at{};
    hipError_t ae = hipFuncGetAttributes(&at, (const void*)&k41);
    printf("k41: arch=%s attr_err=%d staticShared=%zu regs=%d\n",
           p.gcnArchName, (int)ae, at.sharedSizeBytes, (int)at.numRegs);
    k41<<<1, 128>>>(d);
    printf("  launch: %d %s\n", (int)hipGetLastError(), hipGetErrorString(hipGetLastError()));
    hipDeviceSynchronize();
    printf("  sync:   %d %s\n", (int)hipGetLastError(), hipGetErrorString(hipGetLastError()));
    return 0;
}
#ifndef PROBE_NO_MAIN
int main() { return run_probe(); }
#endif
