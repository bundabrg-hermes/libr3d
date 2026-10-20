// probe_l401.hip -- why does a WMMA kernel with ~40 KB static LDS fail to LAUNCH
// (hipErrorNotPermitted 401) on gfx1100? Reports func attributes and launch errors
// for the exact resource shape of r3d_attn_paged_wmma_kernel<128>: static LDS,
// 128 threads, launch_bounds.
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
namespace rw = rocwmma;

// Shape mimic: the real carve is 40992 B static (Q 64x132x2B + S 64x40x4B + P 64x36x2B
// + KV 32x132x2B + meta). Allocate the same size statically and do one WMMA.
__global__ __launch_bounds__(128) void k40(rw::bfloat16_t* o) {
    __shared__ unsigned char raw[40992];
    rw::fragment<rw::accumulator, 16, 16, 16, float> acc;
    rw::fill_fragment(acc, 0.0f);
    rw::fragment<rw::matrix_a, 16, 16, 16, rw::bfloat16_t, rw::row_major> fa;
    rw::fragment<rw::matrix_b, 16, 16, 16, rw::bfloat16_t, rw::col_major> fb;
    rw::fill_fragment(fa, rw::bfloat16_t(1.0f));
    rw::fill_fragment(fb, rw::bfloat16_t(1.0f));
    rw::mma_sync(acc, fa, fb, acc);
    reinterpret_cast<rw::bfloat16_t*>(raw)[threadIdx.x] = rw::bfloat16_t(acc.x[0]);
    __syncthreads();
    if (threadIdx.x == 0) o[0] = reinterpret_cast<rw::bfloat16_t*>(raw)[0];
}
__global__ __launch_bounds__(128) void k32(rw::bfloat16_t* o) {
    __shared__ unsigned char raw[32 * 1024];
    rw::fragment<rw::accumulator, 16, 16, 16, float> acc;
    rw::fill_fragment(acc, 0.0f);
    rw::fragment<rw::matrix_a, 16, 16, 16, rw::bfloat16_t, rw::row_major> fa;
    rw::fragment<rw::matrix_b, 16, 16, 16, rw::bfloat16_t, rw::col_major> fb;
    rw::fill_fragment(fa, rw::bfloat16_t(1.0f));
    rw::fill_fragment(fb, rw::bfloat16_t(1.0f));
    rw::mma_sync(acc, fa, fb, acc);
    reinterpret_cast<rw::bfloat16_t*>(raw)[threadIdx.x] = rw::bfloat16_t(acc.x[0]);
    __syncthreads();
    if (threadIdx.x == 0) o[0] = reinterpret_cast<rw::bfloat16_t*>(raw)[0];
}

template <typename F>
static void tryk(const char* name, F f, rw::bfloat16_t* d) {
    hipFuncAttributes at{};
    hipError_t ae = hipFuncGetAttributes(&at, (const void*)f);
    printf("%s: attr_err=%d staticShared=%zu regs=%d\n",
           name, (int)ae, at.sharedSizeBytes, at.numRegs);
    f<<<1, 128>>>(d);
    printf("  launch: %d %s\n", (int)hipGetLastError(), hipGetErrorString(hipGetLastError()));
    hipDeviceSynchronize();
    printf("  sync:   %d %s\n", (int)hipGetLastError(), hipGetErrorString(hipGetLastError()));
}

int main() {
    int dev = 0; hipDeviceProp_t p{};
    hipGetDevice(&dev); hipGetDeviceProperties(&p, dev);
    printf("arch=%s sharedPerBlock=%zu sharedPerSM=%zu regsPerSM=%d\n",
           p.gcnArchName, p.sharedMemPerBlock, p.sharedMemPerMultiprocessor, p.regsPerMultiprocessor);
    rw::bfloat16_t* d; hipMalloc(&d, 2);
    tryk("k32", k32, d);
    tryk("k40", k40, d);
    return 0;
}
