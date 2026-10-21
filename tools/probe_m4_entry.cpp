/* dlopen harness: real libr3d.so entries, hand-built RadArgs, oracle on host. */
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include "radiance/rad_abi.h"
#include "radiance/rad_types.h"

typedef int (*LaunchFn)(const RadArgs*, RadStream);

static float e4m3h(uint8_t b) {
    uint32_t s = (uint32_t)(b >> 7) << 31;
    uint32_t e = (b >> 3) & 0xfu, m = b & 0x7u; uint32_t u;
    if (e == 0u) { if (m == 0u) { float z; uint32_t zz = s; memcpy(&z,&zz,4); return z; }
                   float f = (float)m / 512.0f; return (b>>7)?-f:f; }
    if (e == 0xfu && m == 0x7u) return NAN;
    u = s | ((e-7u+127u)<<23) | (m<<20); float f; memcpy(&f,&u,4); return f;
}
static float e2m1h(uint8_t n){ static const float V[8]={0,.5f,1,1.5f,2,3,4,6}; return (n&8)?-V[n&7]:V[n&7]; }

int main(int argc, char** argv) {
    int M = argc>1?atoi(argv[1]):1, K = argc>2?atoi(argv[2]):5120, N = argc>3?atoi(argv[3]):16384;
    void* h = dlopen("/work/build13/radiance_home/kernels/libr3d.so", RTLD_NOW);
    if (!h) { printf("dlopen: %s\n", dlerror()); return 1; }
    LaunchFn tile = (LaunchFn)dlsym(h, "r3d_rad_gemm_nt_q");
    LaunchFn gemv = (LaunchFn)dlsym(h, "r3d_rad_gemm_nt_q_mxfp4_decode");
    if (!tile || !gemv) { printf("dlsym fail\n"); return 1; }

    int as_cols=(K+127)/128, bs_cols=(K+31)/32;
    size_t abytes=(size_t)M*K, bbytes=(size_t)N*(K/2), abytes2=(size_t)N*bs_cols;
    uint8_t *A=new uint8_t[abytes], *B=new uint8_t[bbytes], *BS=new uint8_t[abytes2];
    float *AS=new float[(size_t)M*as_cols];
    srand(777);
    for(size_t i=0;i<abytes;i++) A[i]=(uint8_t)(rand()%224)&0x7e;
    for(size_t i=0;i<bbytes;i++) B[i]=(uint8_t)rand();
    for(size_t i=0;i<abytes2;i++) BS[i]=getenv("UNISC")?127:(uint8_t)(115+rand()%25);
    for(int i=0;i<M*as_cols;i++) AS[i]=getenv("UNISA")?0.005f:0.001f+0.01f*(float)(rand()%1000)/1000.f;
    uint8_t *dA,*dB,*dBS; float *dAS; uint16_t *dY1,*dY2;
    hipMalloc((void**)&dA,abytes); hipMalloc((void**)&dB,bbytes); hipMalloc((void**)&dBS,abytes2);
    hipMalloc((void**)&dAS,(size_t)M*as_cols*4); hipMalloc((void**)&dY1,(size_t)M*N*2); hipMalloc((void**)&dY2,(size_t)M*N*2);
    hipMemcpy(dA,A,abytes,hipMemcpyHostToDevice); hipMemcpy(dB,B,bbytes,hipMemcpyHostToDevice);
    hipMemcpy(dBS,BS,abytes2,hipMemcpyHostToDevice); hipMemcpy(dAS,AS,(size_t)M*as_cols*4,hipMemcpyHostToDevice);
    hipMemset(dY1,0xEE,(size_t)M*N*2); hipMemset(dY2,0xEE,(size_t)M*N*2);

    RadTensor t[7] = {};
    t[0]=(RadTensor){dA,RAD_F8E4M3,2,{M,K},{K,1}};
    t[1]=(RadTensor){dAS,RAD_F32,2,{M,as_cols},{as_cols,1}};
    t[2]=(RadTensor){dB,RAD_FP4E2M1,2,{N,K},{K,1}};   /* strides ELEMENTS */
    t[3]=(RadTensor){dBS,RAD_E8M0,2,{N,bs_cols},{bs_cols,1}};
    t[4]=(RadTensor){dY1,RAD_BF16,2,{M,N},{N,1}};
    t[5]=(RadTensor){nullptr,RAD_F32,0,{0},{0}};        /* a_sum absent */
    t[6]=(RadTensor){nullptr,RAD_BF16,0,{0},{0}};       /* b_ref absent */
    RadParam p[] = {
        RAD_STR("dtype","mxfp4a8"), RAD_INT("N",N), RAD_INT("K",K), RAD_INT("group",32),
    };
    RadArgs args = { t, 7, p, (int)(sizeof p/sizeof p[0]), nullptr, 0, nullptr, 0, 0 };

    int rct = tile(&args, 0);
    ((RadTensor*)args.t)[4].data = dY2;                    /* gemv writes the second buffer */
    int rcg = gemv(&args, 0);
    hipError_t le = hipDeviceSynchronize();
    printf("tile rc=%d gemv rc=%d sync=%s\n", rct, rcg, hipGetErrorName(le));
    if (le) return 1;

    uint16_t *Yt=new uint16_t[M*N], *Yg=new uint16_t[M*N];
    hipMemcpy(Yt,dY1,(size_t)M*N*2,hipMemcpyDeviceToHost);
    hipMemcpy(Yg,dY2,(size_t)M*N*2,hipMemcpyDeviceToHost);

    /* oracle over a sample of n's */
    double wt=0,wg=0,tt=0; int wn=-1;
    int step = N/64 > 0 ? N/64 : 1;
    for (int n=0;n<N;n+=step) for (int m=0;m<M;m++) {
        float acc=0;
        for (int g=0; g<bs_cols; g++) {
            float s=0;
            for (int k=0;k<32;k++){
                int kk=g*32+k;
                uint8_t byte=B[(size_t)n*(K/2)+(kk>>1)];
                uint8_t nib=((kk&1)?(byte>>4):(byte&0xf));
                s+=e4m3h(A[(size_t)m*K+kk])*e2m1h(nib);
            }
            uint8_t x=BS[(size_t)n*bs_cols+g];
            float sc=(x==0xff)?NAN:ldexpf(1.0f,(int)x-127);
            acc+=(s*AS[(size_t)m*as_cols+(g>>2)])*sc;
        }
        float ft,fg; { uint32_t u=(uint32_t)Yt[(size_t)m*N+n]<<16; memcpy(&ft,&u,4); }
                   { uint32_t u=(uint32_t)Yg[(size_t)m*N+n]<<16; memcpy(&fg,&u,4); }
        double dref=fabs(acc)>1e-20?fabs(acc):1e-20;
        double dt=fabs(ft-acc)/dref, dg=fabs(fg-acc)/dref, dg2=abs(fg-ft)/dref;
        if(dt>wt){wt=dt;} if(dg>wg){wg=dg;wn=n;} if(dg2>tt)tt=dg2;
        if(getenv("V")&&dg>0.03&&n<2000) printf("m=%d n=%d got=%g tile=%g ref=%g ratio=%g\n",m,n,fg,ft,acc,fg/acc);
    }
    printf("M=%d K=%d N=%d  tile_vs_ref=%g  gemv_vs_ref=%g (n=%d)  gemv_vs_tile=%g\n",M,K,N,wt,wg,wn,tt);
    return wg>0.03?1:0;
}
