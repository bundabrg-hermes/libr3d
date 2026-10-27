// q48_probe_dbg: FORENSICS TEMPLATE. Law 1 (vacuous harness, third costume):
// an early revision of this tool shuffled `g` in the ref leg, so its
// got!=ref=0 was a tautology -- caught desk-side, the ref leg now shuffles
// its own arm `r`; a comparison path must read its OWN arm's state.
// Finding 2 (fold-path codegen, softened by the fixed run): the disassembly
// showed the three lowering forms of `x += partner` (shfl->ds_bpermute, DPP
// row_xmask, LDS oracle) emitted with DIFFERENT v_add_f32 operand orders.
// Empirically the DPP and shfl chains are bit-identical on every input
// tested (got!=ref = 0, including raw sNaN torture bits); the LDS-oracle
// path diverges on ~1.2e-4 of raw-bit iterations, only where the fold's
// running value and partner are BOTH sNaN (NaN payload preference depends
// on the add's emitted operand order). Root cause not chased: production
// rowsum inputs (exp2 sums) are finite by construction, the shfl-vs-DPP
// pair -- the only pair the FNV substitution claims -- is exact. Re-check
// after any toolchain bump (codegen contract, Q42 lesson class).
// isolate the rowsum fold mismatch: per-step three-way on a
// single value, count gr = got!=ref, go = got!=oracle, ro = ref!=oracle,
// dump first 8 failing cases (lane, step, a, ref, ora, got bits).
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>

__device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7eb37b2fu; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
template <int MASK>
__device__ __forceinline__ uint32_t dpp_x(uint32_t p) {
    return __builtin_amdgcn_update_dpp(p, p, 0x160u | MASK, 0xf, 0xf, 0);
}
extern "C" __global__ void dbg(unsigned long long* ctr, int iters) {
    __shared__ float sm[32];
    __shared__ int dump_budget;
    if (threadIdx.x == 0) dump_budget = 8;
    int lane = threadIdx.x & 31;
    __syncthreads();
    unsigned long long gr = 0, go = 0, ro = 0;
    for (int it = 0; it < iters; it++) {
        uint32_t a = hash32((lane * 2654435761u) ^ (it * 40503u) ^ (blockIdx.x * 97u));
        float g = __uint_as_float(a), r = g, o = g;
        #pragma unroll
        for (int m = 1; m <= 8; m <<= 1) {
            uint32_t pg = m == 1 ? dpp_x<1>(__float_as_uint(g))
                        : m == 2 ? dpp_x<2>(__float_as_uint(g))
                        : m == 4 ? dpp_x<4>(__float_as_uint(g))
                                 : dpp_x<8>(__float_as_uint(g));
            float pf = __shfl_xor(r, m);
            sm[lane] = o;
            __syncthreads();
            float po = sm[lane ^ m];
            g += __uint_as_float(pg);
            r += pf;
            o += po;
            __syncthreads();
        }
        uint32_t bg = __float_as_uint(g), br = __float_as_uint(r), bo = __float_as_uint(o);
        gr += (bg != br); go += (bg != bo); ro += (br != bo);
        if ((bg != br || bg != bo) && atomicAdd(&dump_budget, -1) > 0)
            printf("FAIL lane=%2d it=%d a=%08x g=%08x r=%08x o=%08x\n",
                   lane, it, a, bg, br, bo);
    }
    atomicAdd(ctr, gr); atomicAdd(ctr + 1, go); atomicAdd(ctr + 2, ro);
}
int main() {
    unsigned long long* d; hipMalloc(&d, 3 * 8);
    unsigned long long h[3];
    for (uint32_t seed = 0; seed < 1; seed++) {
        hipMemset(d, 0, 24);
        dbg<<<4096, 32>>>(d, 256);
        hipDeviceSynchronize();
        hipMemcpy(h, d, 24, hipMemcpyDeviceToHost);
        printf("got!=ref=%llu got!=ora=%llu ref!=ora=%llu\n", h[0], h[1], h[2]);
    }
    return 0;
}
