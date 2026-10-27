// q49_diag2.hip -- full gated pipeline clone (vec arm restructure) vs scalar arm,
// incl. inf groups + NaN encode; plus a raw operand-order experiment for NaN*0.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
__device__ __forceinline__ float r3d_exp_f(float x) { return (float)exp((double)x); }
__device__ __forceinline__ float r3d_sigmoid(float x) { return 1.0f / (1.0f + r3d_exp_f(-x)); }
__device__ __forceinline__ float r3d_bf16_to_f32(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }
__device__ __forceinline__ uint16_t r3d_f32_to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ uint8_t e4m3(float f) {   // r3d_band's encoder (copied)
    const uint32_t u = __float_as_uint(f);
    const uint8_t  s = (uint8_t)((u >> 24) & 0x80u);
    const uint32_t a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return (uint8_t)(s | 0x7fu);
    if (a >= 0x43e00000u) return (uint8_t)(s | 0x7eu);
    const int e = (int)(a >> 23) - 127;
    uint32_t m = a & 0x7fffffu;
    if (e < -10) return s;
    if (e < -6) { const int sh = -6 - e + 20; m |= 0x800000u;
        const uint32_t lo = m & ((1u << sh) - 1u); uint32_t q = m >> sh;
        const uint32_t half = 1u << (sh - 1);
        if (lo > half || (lo == half && (q & 1u))) ++q; return (uint8_t)(s | q); }
    uint32_t q = m >> 20; const uint32_t lo = m & 0xfffffu; int ee;
    if (lo > 0x80000u || (lo == 0x80000u && (q & 1u))) ++q;
    ee = e + 7; if (q == 0x8u) { q = 0; ++ee; }
    if (ee > 15 || (ee == 15 && q >= 7u)) return (uint8_t)(s | 0x7eu);
    return (uint8_t)(s | ((uint32_t)ee << 3) | q);
}
__device__ __forceinline__ float gated_v(float gt, float up) {
    const float sg = r3d_sigmoid(gt);
    return r3d_bf16_to_f32(r3d_f32_to_bf16((gt * sg) * up));
}
#define NG 128
__global__ void pipe_scalar(const uint16_t* gu, uint8_t* q, float* sc_out, int n) {  // one block per row, group loop
    __shared__ float red[8]; __shared__ float s_sc;
    for (int gi = 0; gi < n / NG; ++gi) {
        const int k0 = gi * NG;
        float amax = 0.0f;
        for (int i = threadIdx.x; i < NG; i += 256) {
            const float m = fabsf(gated_v(r3d_bf16_to_f32(gu[i + k0]), r3d_bf16_to_f32(gu[n + i + k0])));
            if (m > amax) amax = m;
        }
        for (int o = 16; o; o >>= 1) { const float p = __shfl_down_sync((unsigned long long)~0ull, amax, o); if (p > amax) amax = p; }
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
        __syncthreads();
        if (threadIdx.x == 0) { float mx = 0.0f; for (int w2 = 0; w2 < 8; ++w2) if (red[w2] > mx) mx = red[w2];
            s_sc = mx > 0.0f ? mx * (1.0f / 448.0f) : 1.0f; sc_out[gi] = s_sc; }
        __syncthreads();
        const float inv = 1.0f / s_sc;
        for (int i = threadIdx.x; i < NG; i += 256) {
            float h = gated_v(r3d_bf16_to_f32(gu[i + k0]), r3d_bf16_to_f32(gu[n + i + k0]));
            float c = h * inv;
            if (c >  448.0f) c =  448.0f;
            if (c < -448.0f) c = -448.0f;
            q[k0 + i] = e4m3(c);
        }
        __syncthreads();
    }
}
__global__ void pipe_vec(const uint16_t* gu, uint8_t* q, float* sc_out, int n) {     // subgroup restructure
    const uint32_t cpg = NG / 8, ngrp = n / NG, gpb = 256 / cpg;
    const uint32_t sub = threadIdx.x % cpg, wgrp = threadIdx.x / cpg;
    const uint32_t sbase = threadIdx.x & ~(cpg - 1u);
    const uint32_t mask = (((1u << cpg) - 1u) << sbase);
    for (uint32_t base = 0; base < ngrp; base += gpb) {
        const uint32_t gi = base + wgrp;
        const uint32_t c = gi * cpg + sub;
        const uint4 gv = ((const uint4*)gu)[c];
        const uint4 uv = ((const uint4*)(gu + n))[c];
        const uint32_t* gp = &gv.x; const uint32_t* up_ = &uv.x;
        float h[8]; float amax = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            h[2*k]   = gated_v(r3d_bf16_to_f32((uint16_t)(gp[k] & 0xffffu)), r3d_bf16_to_f32((uint16_t)(up_[k] & 0xffffu)));
            h[2*k+1] = gated_v(r3d_bf16_to_f32((uint16_t)(gp[k] >> 16)),     r3d_bf16_to_f32((uint16_t)(up_[k] >> 16)));
            const float m0 = fabsf(h[2*k]), m1 = fabsf(h[2*k+1]);
            if (m0 > amax) amax = m0; if (m1 > amax) amax = m1;
        }
        for (uint32_t off = cpg / 2; off; off >>= 1) {
            const float p = __shfl_xor_sync((unsigned long long)mask, amax, (int)off);
            if (p > amax) amax = p;
        }
        const float sc = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
        if (sub == 0) sc_out[gi] = sc;
        const float inv = 1.0f / sc;
        uint8_t code[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float t = h[k] * inv;
            if (t >  448.0f) t =  448.0f;
            if (t < -448.0f) t = -448.0f;
            code[k] = e4m3(t);
        }
        memcpy(q + (size_t)c * 8, code, 8);
    }
}
__global__ void mulorder(float* out, const float* nan_in) {   // which operand's sign wins
    const float n = nan_in[0], z = 0.0f;
    out[0] = n * z; out[1] = z * n;                       // a*0 vs 0*a
    out[2] = __uint_as_float(__float_as_uint(n * z) ^ 0u);
}
int main() {
    const int n = 4096; uint16_t* h = new uint16_t[2 * n];
    uint32_t st = 1; auto rnd = [&]{ st ^= st<<13; st ^= st>>17; st ^= st<<5; return st; };
    for (int i = 0; i < 2 * n; ++i) {
        const uint32_t r = rnd();
        if ((r & 0xffffu) < 256u) h[i] = (uint16_t)((r & 0x8000u) | 0x7FC0u | (r & 0x3fu));           // qNaN any sign
        else if ((r & 0xffffu) < 384u) h[i] = (uint16_t)((r & 0x8000u) | 0x7F80u);                     // +-inf
        else h[i] = (uint16_t)(((r & 0x8000u) << 1) | ((r >> 17) & 0x8000u) | (r & 0x7fffu));
    }
    uint16_t* dg; uint8_t *dq1, *dq2; float *ds1, *ds2;
    hipMalloc(&dg, 2*n*2); hipMalloc(&dq1, n); hipMalloc(&dq2, n); hipMalloc(&ds1, n/NG*4); hipMalloc(&ds2, n/NG*4);
    hipMemcpy(dg, h, 2*n*2, hipMemcpyHostToDevice);
    pipe_scalar<<<1,256>>>(dg, dq1, ds1, n);
    pipe_vec<<<1,256>>>(dg, dq2, ds2, n);
    uint8_t o1[4096], o2[4096]; float s1[32], s2[32];
    hipMemcpy(o1, dq1, n, hipMemcpyDeviceToHost); hipMemcpy(o2, dq2, n, hipMemcpyDeviceToHost);
    hipMemcpy(s1, ds1, 32*4, hipMemcpyDeviceToHost); hipMemcpy(s2, ds2, 32*4, hipMemcpyDeviceToHost);
    int nd = 0, sd = 0;
    for (int i = 0; i < n; ++i) if (o1[i] != o2[i]) {
        if (nd < 6) printf("i=%d gate=%04x up=%04x q_sca=%02x q_vec=%02x\n", i, h[i], h[n+i], o1[i], o2[i]);
        ++nd; }
    for (int i = 0; i < n/NG; ++i) if (s1[i] != s2[i]) ++sd;
    printf("DIFFS %d/%d scalediffs %d\n", nd, n, sd);
    float nanv; float* dn; float* dout; hipMalloc(&dn, 4); hipMalloc(&dout, 12);
    { union{uint32_t u; float f;} cv; cv.u=0x7fc00000u; nanv=cv.f; } hipMemcpy(dn, &nanv, 4, hipMemcpyHostToDevice);
    mulorder<<<1,1>>>(dout, dn);
    float o[3]; hipMemcpy(o, dout, 12, hipMemcpyDeviceToHost);
    { union{float f; uint32_t u;} c1, c2; c1.f=o[0]; c2.f=o[1];
      printf("nan*0 = %08x, 0*nan = %08x\n", c1.u, c2.u); }
    return 0;
}
