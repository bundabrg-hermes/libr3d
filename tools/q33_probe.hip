// q33_probe: the Q33 card gate -- lane-exact three-way adjudication of the
// bf16 GEMV butterfly fold on gfx1100 BEFORE any A/B (the q24_fullprobe law).
//   ref  = __shfl_xor(v, o)               (the shipped v151 butterfly stage)
//   ora  = smem[lane ^ o]                 (independent LDS oracle)
//   got  = permlanex16 (o=16) / DPP row_xmask (o=8/4/2/1)
// Plus the EXACT shipped fold shape: the bf16 descending butterfly 16,8,4,2,1
// (r3d_gemm_bf16.hip gemv_nt_bf16_kernel) run at lever 0/1/2 across a large
// lane-op sweep; all three levers must agree bit-for-bit (partner value +
// same add order, -ffp-contract=off f32 adds). Q24's known failure class:
// the swapped permlanex16 SGPR pairing silently gathers the wrong partner --
// no crash, no build error. Deterministic hashes, atomicAdd failure counters;
// PROBE-BITEXACT iff zero.
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>

#define NWAVE 4096
__device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7eb37b2fu; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
// exact mirrors of the shipped Q33 helpers (src/r3d_gemm_bf16.hip)
template <int MASK>
__device__ __forceinline__ uint32_t dpp_x(uint32_t p) {
    return __builtin_amdgcn_update_dpp(p, p, 0x160u | MASK, 0xf, 0xf, 0);
}
__device__ __forceinline__ uint32_t perm_x16(uint32_t v) {
    return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xFEDCBA98u, false, false);
}
// the shipped butterfly fold at lever dpp (0 = verbatim shfl chain)
__device__ __forceinline__ float fold_bfly_f32(float c, const int dpp) {
    if (dpp) {
        c += dpp >= 2 ? __uint_as_float(perm_x16(__float_as_uint(c)))
                      : __shfl_xor(c, 16);
        c += __uint_as_float(dpp_x<8>(__float_as_uint(c)));
        c += __uint_as_float(dpp_x<4>(__float_as_uint(c)));
        c += __uint_as_float(dpp_x<2>(__float_as_uint(c)));
        c += __uint_as_float(dpp_x<1>(__float_as_uint(c)));
        return c;
    }
    c += __shfl_xor_sync(0xffffffffffffffffull, c, 16);
    c += __shfl_xor_sync(0xffffffffffffffffull, c, 8);
    c += __shfl_xor_sync(0xffffffffffffffffull, c, 4);
    c += __shfl_xor_sync(0xffffffffffffffffull, c, 2);
    c += __shfl_xor_sync(0xffffffffffffffffull, c, 1);
    return c;
}

extern "C" __global__ void probe_bfly(uint32_t seed, unsigned long long* fail_xor,
                                      unsigned long long* fail_row,
                                      unsigned long long* fail_chain, int iters) {
    __shared__ uint32_t sm[32];
    int lane = threadIdx.x & 31;
    uint32_t acc_xor = 0, acc_row = 0, acc_chain = 0;
    for (int it = 0; it < iters; it++) {
        uint32_t v0 = hash32((lane * 2654435761u) ^ (seed + it * 40503u)
                             ^ (blockIdx.x * 97u));
        // x16 site three-way: shfl vs LDS oracle vs permlanex16
        uint32_t ref = __shfl_xor(v0, 16);
        uint32_t got = perm_x16(v0);
        sm[lane] = v0;
        __syncthreads();
        uint32_t ora = sm[lane ^ 16];
        acc_xor += (ref != ora) + (got != ora);
        if (blockIdx.x == 0 && it == 0)
            printf("lane %2u v0=%08x ref=%08x ora=%08x got=%08x\n",
                   lane, v0, ref, ora, got);
        __syncthreads();
        // row sites (8/4/2/1) three-way: shfl vs LDS oracle vs DPP row_xmask
        #pragma unroll
        for (int o = 1; o <= 8; o <<= 1) {
            sm[lane] = v0;
            __syncthreads();
            uint32_t r = __shfl_xor(v0, o);
            uint32_t g = (o == 8 ? dpp_x<8>(v0) : o == 4 ? dpp_x<4>(v0)
                     : o == 2 ? dpp_x<2>(v0) : dpp_x<1>(v0));
            uint32_t q = sm[lane ^ o];
            acc_row += (r != q) + (g != q);
            __syncthreads();
        }
        // fold-chain bit-identity on the SHIPPED shape: lever 0 vs 1 vs 2
        float f0 = __uint_as_float(v0), f1 = f0, f2 = f0;
        f0 = fold_bfly_f32(f0, 0); f1 = fold_bfly_f32(f1, 1); f2 = fold_bfly_f32(f2, 2);
        acc_chain += (__float_as_uint(f0) != __float_as_uint(f1))
                   + (__float_as_uint(f0) != __float_as_uint(f2));
    }
    if (acc_xor)   atomicAdd(fail_xor, acc_xor);
    if (acc_row)   atomicAdd(fail_row, acc_row);
    if (acc_chain) atomicAdd(fail_chain, acc_chain);
}

int main() {
    unsigned long long* d;
    hipMalloc(&d, 3 * sizeof(unsigned long long));
    unsigned long long h[3];
    unsigned long long tot[3] = {0, 0, 0};
    const int iters = 256;
    for (uint32_t seed = 0; seed < 8; seed++) {
        hipMemset(d, 0, 3 * sizeof(unsigned long long));
        probe_bfly<<<NWAVE, 32>>>(seed, d, d + 1, d + 2, iters);
        hipDeviceSynchronize();
        hipMemcpy(h, d, 3 * sizeof(unsigned long long), hipMemcpyDeviceToHost);
        for (int i = 0; i < 3; i++) tot[i] += h[i];
        printf("seed %u: xor-fail=%llu row-fail=%llu chain-fail=%llu\n",
               seed, h[0], h[1], h[2]);
    }
    unsigned long long lanes = (unsigned long long)8 * NWAVE * 32 * iters;
    printf("lane-ops=%llu per site-class (x16 + 4 row + 1 chain shape)\n", lanes);
    printf(tot[0] + tot[1] + tot[2] == 0 ? "PROBE-BITEXACT\n"
           : "PROBE-FAIL %llu/%llu/%llu\n", tot[0], tot[1], tot[2]);
    return tot[0] + tot[1] + tot[2];
}
