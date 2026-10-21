/* standalone probe for the mxfp4 decode GEMV (throwaway; delete after use) */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>

namespace {
constexpr int kThreads = 256;
__device__ __forceinline__ uint16_t f32_to_bf16_m4(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float fp8e4m3_gemv(uint8_t b) {
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
__device__ const uint16_t kE2M1[16] = {
    0x0000u, 0x3F00u, 0x3F80u, 0x3FC0u, 0x4000u, 0x4040u, 0x4080u, 0x40C0u,
    0x8000u, 0xBF00u, 0xBF80u, 0xBFC0u, 0xC000u, 0xC040u, 0xC080u, 0xC0C0u,
};
template <int MT>
__global__ __launch_bounds__(kThreads) void gemm_nt_mxfp4_gemv_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, uint16_t* __restrict__ Y,
    const float* __restrict__ AS, const uint8_t* __restrict__ BS,
    int M, int N, int K,
    int64_t a_pitch, int64_t b_pitch,        /* BYTES (nibbles: K/2 of them) */
    int64_t y_pitch, int64_t as_pitch, int64_t bs_pitch)
{
    __shared__ __align__(16) float xs[MT][256];
    __shared__ float L16[16];                 /* e2m1 nibble -> exact f32 (kE2M1 << 16) */

    if (threadIdx.x < 16)
        L16[threadIdx.x] = __uint_as_float((uint32_t)kE2M1[threadIdx.x] << 16);

    const int w    = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int r    = blockIdx.x * (kThreads / 32) + w;
    const int rr   = r < N ? r : N - 1;               /* dead warps load a live row */
    const int live = M < MT ? M : MT;

    const uint8_t* crow = B + (int64_t)rr * b_pitch;
    const uint8_t* brow = BS + (int64_t)rr * bs_pitch;   /* b_scale is per weight row */

    for (int i = threadIdx.x; i < MT * 256; i += kThreads)
        xs[i >> 8][i & 255] = 0.0f;
    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;
    __syncthreads();                            /* LUT and xs visible */

    const int n256 = K >> 8;                      /* K % 128 == 0 is the row gate */
    const bool partial = (K & 255) != 0;          /* one trailing 128-wide span */
    for (int it = 0; it < n256 + (partial ? 1 : 0); ++it) {
        const int kb   = it << 8;
        const int rem  = (it < n256) ? 256 : 128;
        const int shift = (rem == 256) ? 8 : 7;
        for (int i = threadIdx.x; i < live << shift; i += kThreads) {
            const int m = i >> shift, k = i & (rem - 1);
            xs[m][k] = fp8e4m3_gemv(A[(int64_t)m * a_pitch + kb + k]);
        }
        __syncthreads();

        /* One aligned dword per lane: nibbles kb + lane*8 + 0..7. Nibble k lives in byte
         * k>>1 (the tile's rule), so the dword is at byte (kb>>1) + lane*4 -- indexing
         * the packed row by k DOUBLES the stride, the nibble form of the byte-pitch trap. */
        const bool act = (lane << 3) < rem;
        uint32_t b4 = 0u;
        if (act) b4 = *(const uint32_t*)(crow + (kb >> 1) + (lane << 2));

        const int sg = lane >> 2;                 /* this quad's group inside the span */
        const bool qact = sg < (rem >> 5);
        const int  g32 = (it << 3) + (qact ? sg : (rem >> 5) - 1);
        float sc = 0.0f;
        if (qact) {
            const uint8_t x = brow[g32];
            sc = (x == 0xffu) ? __uint_as_float(0x7fc00000u)
               : (x == 0x00u) ? 0x1p-127f
                              : __uint_as_float((uint32_t)x << 23);
        }

        float sub[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            float s = 0.0f;
            if (act) {
                const float4* xp = (const float4*)&xs[m][lane << 3];
                const float4 v0 = xp[0], v1 = xp[1];
                const float x8[8] = { v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w };
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const uint32_t nib = (b4 >> (4 * t)) & 15u;   /* low nibble = even k */
                    s = __fmaf_rn(x8[t], L16[nib], s);            /* product exact in f32 */
                }
            }
            sub[m] = s;
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
#pragma unroll
            for (int off = 1; off <= 2; off <<= 1)             /* fold the quad (k chain) */
                sub[m] += __shfl_xor(sub[m], off);
            /* Dead quads join every shfl below as exact zeros -- skipping the collective
             * with `continue` would leave offsets 4/8/16 facing absent partners. Their
             * sa/sc reads are clamped in bounds and discarded. */
            const float sa = AS[(int64_t)m * as_pitch + (g32 >> 2)];
            float c = qact ? (sub[m] * sa) * sc : 0.0f;         /* quant_dot's two-multiply order */
#pragma unroll
            for (int off = 4; off; off <<= 1)                  /* fold the eight group terms */
                c += __shfl_xor(c, off);
            acc[m] += c;
        }
        __syncthreads();                                       /* xs is rewritten next span */
    }

    if (lane == 0 && r < N) {
#pragma unroll
        for (int m = 0; m < MT; ++m)
            if (m < live) Y[(int64_t)m * y_pitch + r] = f32_to_bf16_m4(acc[m]);
    }
}

}

static float e4m3h(uint8_t b) { // host decode, same semantics
    uint32_t s = (uint32_t)(b >> 7) << 31;
    uint32_t e = (b >> 3) & 0xfu, m = b & 0x7u; uint32_t u;
    if (e == 0u) { if (m==0u) { float z; uint32_t zz=s; memcpy(&z,&zz,4); return z; } float f=(float)m/512.0f; return (b>>7)?-f:f; }
    if (e==0xfu && m==0x7u) return NAN;
    u = s | ((e-7u+127u)<<23) | (m<<20); float f; memcpy(&f,&u,4); return f;
}
static float __uint_as_float_host(uint32_t u){ float f; memcpy(&f,&u,4); return f; }
static float e2m1h(uint8_t n){ static const float V[8]={0,.5f,1,1.5f,2,3,4,6}; return (n&8)?-V[n&7]:V[n&7]; }

int main(int argc, char** argv) {
    int M = argc>1?atoi(argv[1]):1, K = argc>2?atoi(argv[2]):512, N = argc>3?atoi(argv[3]):5;
    int MT = M>=8?8:M>=4?4:M>=2?2:1;
    int as_cols=(K+127)/128, bs_cols=(K+31)/32;
    size_t abytes=(size_t)M*K, bbytes=(size_t)N*(K/2), ybytes=(size_t)M*N*2;
    uint8_t *A=new uint8_t[abytes], *B=new uint8_t[bbytes], *BS=new uint8_t[(size_t)N*bs_cols];
    float *AS=new float[(size_t)M*as_cols]; uint16_t *Y=new uint16_t[M*N];
    srand(777);
    for(size_t i=0;i<abytes;i++) A[i]=(uint8_t)(rand()%224)&0x7e;          // avoid NaN byte 0x7f
    for(size_t i=0;i<bbytes;i++) B[i]=(uint8_t)(rand()&0xff);
    for(int i=0;i<N*bs_cols;i++) BS[i]=(uint8_t)(115+rand()%25);      // ~2^-7..2^7
    for(int i=0;i<M*as_cols;i++) AS[i]=0.001f+0.01f*(float)(rand()%1000)/1000.f;
    uint8_t *dA,*dB,*dBS; float *dAS; uint16_t *dY;
    hipMalloc(&dA,abytes); hipMalloc(&dB,bbytes); hipMalloc(&dBS,(size_t)N*bs_cols);
    hipMalloc(&dAS,(size_t)M*as_cols*4); hipMalloc(&dY,ybytes);
    hipMemcpy(dA,A,abytes,hipMemcpyHostToDevice); hipMemcpy(dB,B,bbytes,hipMemcpyHostToDevice);
    hipMemcpy(dBS,BS,(size_t)N*bs_cols,hipMemcpyHostToDevice);
    hipMemcpy(dAS,AS,(size_t)M*as_cols*4,hipMemcpyHostToDevice);
    dim3 grid((N+7)/8);
    #define L(MT_) gemm_nt_mxfp4_gemv_kernel<MT_><<<grid, dim3(kThreads), 0, 0>>>(dA,dB,dY,dAS,dBS,M,N,K,(int64_t)K,(int64_t)(K/2),(int64_t)N,(int64_t)as_cols,(int64_t)bs_cols)
    switch(MT){case 1: L(1);break;case 2: L(2);break;case 4: L(4);break;default: L(8);break;}
    hipError_t le=hipDeviceSynchronize();
    if(le!=hipSuccess){printf("LAUNCH/RUN FAIL %s\n",hipGetErrorName(le));return 1;}
    hipMemcpy(Y,dY,ybytes,hipMemcpyDeviceToHost);
    double worst=0; int wm=-1,wn=-1;
    for(int n=0;n<N;n++) for(int m=0;m<M;m++){
        float acc=0;
        for(int g=0;g<bs_cols;g++){
            double sub=0; // ascending fp32 chain in float
            float s=0;
            for(int k=0;k<32 && g*32+k<K;k++){
                int kk=g*32+k;
                float x=e4m3h(A[(size_t)m*K+kk]);
                uint8_t nib=((kk&1)?(B[(size_t)n*(K/2)+(kk>>1)]>>4):(B[(size_t)n*(K/2)+(kk>>1)]&0xf));
                s+=x*e2m1h(nib);
            }
            float sc=(BS[(size_t)n*bs_cols+g]==0xff)?NAN:(BS[(size_t)n*bs_cols+g]==0)?ldexpf(1,-127):ldexpf(1,(int)BS[(size_t)n*bs_cols+g]-127);
            acc+=(s*AS[(size_t)m*as_cols+(g>>2)])*sc;
        }
        float got; { uint32_t u=(uint32_t)Y[(size_t)m*N+n]<<16; memcpy(&got,&u,4); }
        float d=fabsf(got-acc), r=d/(fabsf(acc)>1e-30f?fabsf(acc):1e-30f);
        if(r>worst){worst=r;wm=m;wn=n;}
        if(getenv("V")) printf("m=%d n=%d got=%g ref=%g\n",m,n,got,acc);
    }
    printf("M=%d K=%d N=%d worst rel=%g at m=%d n=%d\n",M,K,N,worst,wm,wn);
    return worst>0.03?1:0;
}
