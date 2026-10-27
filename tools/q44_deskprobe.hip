/* q44_deskprobe.hip -- Q44 desk ISA-schedule probe standalone (branch card/q44-deskprobe).
 *
 * Two MT-parameterized GEMV inner loops over the IDENTICAL data layout and the
 * IDENTICAL per-group math (same sudot4 chain, same SG/AS/sc scaling, same bf16
 * flush -- every byte reach relation preserved), differing ONLY in fold topology:
 *
 *  FORM A -- the shipped q8x GEMV VERBATIM (src/r3d_gemm_mxfp4.hip:938-1059 region:
 *            quad-owns-group + q8x_fold_i32 + q8x_fold_f32 interleaved per group
 *            per m inside the K loop; both pf arms and both nt legs kept so the
 *            static census matches the shipped binary class-for-class: Q35 law
 *            ds_bpermute 12*MT / permlane 2*MT / DPP 8*MT).
 *
 *  FORM B -- the Q44 proposed lane map (donor: hipfire claude_q8x lane-owns-block
 *            + transposed tail reduce, /tmp/sudot4work/claude_q8x.hip:79-150):
 *            lane t OWNS one full 32-code group per iteration (iteration covers
 *            8 136B blocks = 32 groups = 1024 codes); the int dot completes IN
 *            LANE (exact under int associativity -- no fold_i32 at all), the
 *            scale containment is per-lane (SG/AS/sc all indexed by the lane's
 *            own g), and the wave reduces ONCE at row end (single tail reduce,
 *            DPP kit arm / shfl arm under the same runtime `dpp` lever shape).
 *            Cross-lane ops inside the K loop: ZERO.
 *
 * Bit-identity class: Form A == shipped exactly. Form B's f32 tree differs
 * (per-lane iteration-outer accumulation + 32-way butterfly vs the shipped
 * per-iteration 8-group fold tree) -- this is the row's STEP 2 class (rel_l2/
 * KLD), not step 1; see q44_report.md. K contract: shipped requires K%256==0
 * (no tail loop); Form B requires K%1024==0 for the probe (card session peels
 * the block tail fold-free; measured here: nothing -- the tail loop codegen is
 * schedule-irrelevant to the in-loop proxies).
 *
 * Saveexec audit (window-46 law): every fold here is full-exec -- the m>=live
 * break is BLOCK-UNIFORM (live=min(M,MT), all lanes identical), and the Form B
 * tail reduce is outside every predicate. No fold result is consumed under a
 * lane-divergent select in either form => no vbitmerge guard needed by
 * construction; the ISA audit verifies (tools/q44_schedule_diff.sh step 4).
 *
 * Compile (container radiance-dev:v1.11.0, build-TU-verbatim flags, gfx1100 leg):
 *   /opt/rocm/core-10.1/lib/llvm/bin/clang++ -x hip -D__HIP_PLATFORM_AMD__=1 \
 *     -O3 -DNDEBUG -std=gnu++20 --offload-arch=gfx1100 --no-offload-compress \
 *     -fPIC -ffp-contract=off -g0 -mllvm=-amdgpu-enable-vopd=false \
 *     -mno-wavefrontsize64 -c q44_deskprobe.hip -o q44_deskprobe.o
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstring>

/* ---- verbatim helpers (r3d_mxfp4_q8x.h / r3d_gemm_mxfp4.hip) ---- */
__device__ __host__ inline float r3d_e8m0_sc(uint8_t x) {
    uint32_t u = 0x7fc00000u;
    if (x != 0xffu) u = (x == 0x00u) ? 0x00400000u : ((uint32_t)x << 23);
    float f;
    unsigned char* d = (unsigned char*)&f;
    const unsigned char* s = (const unsigned char*)&u;
    for (int i = 0; i < 4; ++i) d[i] = s[i];
    return f;
}
__device__ __forceinline__ uint16_t f32_to_bf16_m4(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ uint32_t q8x_perm_x16(uint32_t v) {
    return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xFEDCBA98u, false, false);
}
__device__ __forceinline__ int32_t q8x_fold_i32(int32_t p, const int dpp) {
    if (dpp) {
        p += __builtin_amdgcn_update_dpp(p, p, 0x161, 0xf, 0xf, 0);
        p += __builtin_amdgcn_update_dpp(p, p, 0x162, 0xf, 0xf, 0);
        return p;
    }
    p += __shfl_xor(p, 1);
    p += __shfl_xor(p, 2);
    return p;
}
__device__ __forceinline__ float q8x_fold_f32(float c, const int dpp) {
    if (dpp) {
        c += __uint_as_float(__builtin_amdgcn_update_dpp(
            __float_as_uint(c), __float_as_uint(c), 0x164, 0xf, 0xf, 0));
        c += __uint_as_float(__builtin_amdgcn_update_dpp(
            __float_as_uint(c), __float_as_uint(c), 0x168, 0xf, 0xf, 0));
        c += dpp >= 2 ? __uint_as_float(q8x_perm_x16(__float_as_uint(c)))
                      : __shfl_xor(c, 16);
        return c;
    }
    c += __shfl_xor(c, 4);
    c += __shfl_xor(c, 8);
    c += __shfl_xor(c, 16);
    return c;
}

/* ============================ FORM A (shipped verbatim) ============================ */
template <int MT>
__global__ __launch_bounds__(32, 16) void q44_formA_kernel(
    const uint8_t* __restrict__ S, const int8_t* __restrict__ X, const float* __restrict__ SG,
    const float* __restrict__ AS, uint16_t* __restrict__ Y,
    int M, int N, int K, int64_t y_pitch, int64_t as_pitch, int nt, int pf, int dpp)
{
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    const int q = t >> 2;
    const int nstep = K >> 8;
    const int live = M < MT ? M : MT;
    const uint8_t* crow = S + (int64_t)r * ((int64_t)(K >> 7) * 136);

    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

    const int hi8 = (t >> 4) * 136;
    const int lo8 = (t & 15) << 3;

    if (pf >= 2) {
        uint2 wa, wb;
        uint8_t sa, sb;
        auto ldw = [&](int it, uint2& w, uint8_t& sc_b) {
            const uint8_t* sp = crow + ((int64_t)it << 1) * 136;
            if (nt) { const uint64_t raw = __builtin_nontemporal_load(
                                (const uint64_t*)(sp + hi8 + 8 + lo8));
                      __builtin_memcpy(&w, &raw, 8); }
            else      w = *(const uint2*)(sp + hi8 + 8 + lo8);
            sc_b = sp[((q >> 2) * 136) + (q & 3)];
        };
        if (nstep > 0) { ldw(0, wa, sa); if (nstep > 1) ldw(1, wb, sb); }
        for (int it = 0; it < nstep; ++it) {
            uint2 wc;
            uint8_t sbyte;
            if (it & 1) { wc = wb; sbyte = sb;
                          if (it + 2 < nstep) ldw(it + 2, wb, sb); }
            else        { wc = wa; sbyte = sa;
                          if (it + 2 < nstep) ldw(it + 2, wa, sa); }
            const int g = (it << 3) + q;
            const float sc = r3d_e8m0_sc(sbyte);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m >= live) break;
                const uint2 xv = *(const uint2*)(X + (int64_t)m * K + (it << 8) + (t << 3));
                int32_t p = __builtin_amdgcn_sudot4(1, (int)xv.x, 1, (int)wc.x, 0, 0);
                p = __builtin_amdgcn_sudot4(1, (int)xv.y, 1, (int)wc.y, p, 0);
                p = q8x_fold_i32(p, dpp);
                float c = (float)p * SG[(int64_t)m * (K >> 5) + g]
                                * AS[(int64_t)m * as_pitch + (g >> 2)] * sc * 0.5f;
                c = q8x_fold_f32(c, dpp);
                acc[m] += c;
            }
        }
    } else {
    for (int it = 0; it < nstep; ++it) {
        const uint8_t* sp = crow + ((int64_t)it << 1) * 136;
        uint2 wc;
        if (nt) { const uint64_t raw = __builtin_nontemporal_load(
                            (const uint64_t*)(sp + hi8 + 8 + lo8));
                  __builtin_memcpy(&wc, &raw, 8); }
        else      wc = *(const uint2*)(sp + hi8 + 8 + lo8);
        const int g = (it << 3) + q;
        const float sc = r3d_e8m0_sc(sp[((q >> 2) * 136) + (q & 3)]);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m >= live) break;
            const uint2 xv = *(const uint2*)(X + (int64_t)m * K + (it << 8) + (t << 3));
            int32_t p = __builtin_amdgcn_sudot4(1, (int)xv.x, 1, (int)wc.x, 0, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xv.y, 1, (int)wc.y, p, 0);
            p = q8x_fold_i32(p, dpp);
            float c = (float)p * SG[(int64_t)m * (K >> 5) + g]
                            * AS[(int64_t)m * as_pitch + (g >> 2)] * sc * 0.5f;
            c = q8x_fold_f32(c, dpp);
            acc[m] += c;
        }
    }
    }
    if (t == 0) {
#pragma unroll
        for (int m = 0; m < MT; ++m)
            if (m < live) Y[(int64_t)m * y_pitch + r] = f32_to_bf16_m4(acc[m]);
    }
}

/* ==================== FORM B (proposed: lane-owns-group, tail reduce) ====================
 * Lane t owns group g = (it<<5)+t outright: 32 weight bytes + own scale byte;
 * the dot completes in-lane (int, EXACT under associativity), scaling is
 * per-lane contained, acc[m] += c with NO cross-lane traffic in the K loop.
 * Same address algebra as shipped: block(8it + t>>2), sub-block t&3 -- the
 * (group -> bytes, scale) relation is byte-for-byte the shipped one. */
template <int MT>
__global__ __launch_bounds__(32, 16) void q44_formB_kernel(
    const uint8_t* __restrict__ S, const int8_t* __restrict__ X, const float* __restrict__ SG,
    const float* __restrict__ AS, uint16_t* __restrict__ Y,
    int M, int N, int K, int64_t y_pitch, int64_t as_pitch, int nt, int pf, int dpp)
{
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    const int nstep = K >> 10;                /* 32 groups / iteration */
    const int live = M < MT ? M : MT;
    const uint8_t* crow = S + (int64_t)r * ((int64_t)(K >> 7) * 136);

    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

    const int wb = t >> 2;                    /* owning 136B block within the 8-span */
    const int ws = (t & 3) * 32;              /* sub-block byte offset */
    for (int it = 0; it < nstep; ++it) {
        const uint8_t* blk = crow + ((int64_t)it << 3) * 136;
        const uint8_t* wp = blk + wb * 136 + 8 + ws;
        uint4 wa, wbv;
        if (nt) {
            const uint64_t r0 = __builtin_nontemporal_load((const uint64_t*)wp);
            const uint64_t r1 = __builtin_nontemporal_load((const uint64_t*)(wp + 8));
            const uint64_t r2 = __builtin_nontemporal_load((const uint64_t*)(wp + 16));
            const uint64_t r3 = __builtin_nontemporal_load((const uint64_t*)(wp + 24));
            wa = make_uint4((uint32_t)r0, (uint32_t)(r0 >> 32), (uint32_t)r1, (uint32_t)(r1 >> 32));
            wbv = make_uint4((uint32_t)r2, (uint32_t)(r2 >> 32), (uint32_t)r3, (uint32_t)(r3 >> 32));
        } else {
            wa = *(const uint4*)wp;
            wbv = *(const uint4*)(wp + 16);
        }
        const int g = (it << 5) + t;          /* the LANE owns the 32-group */
        const float sc = r3d_e8m0_sc(blk[wb * 136 + (t & 3)]);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m >= live) break;             /* block-uniform: every lane skips */
            const int8_t* xp = X + (int64_t)m * K + ((int64_t)it << 10) + t * 32;
            const uint4 xa = *(const uint4*)xp;
            const uint4 xb = *(const uint4*)(xp + 16);
            int32_t p = __builtin_amdgcn_sudot4(1, (int)xa.x, 1, (int)wa.x, 0, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xa.y, 1, (int)wa.y, p, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xa.z, 1, (int)wa.z, p, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xa.w, 1, (int)wa.w, p, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xb.x, 1, (int)wbv.x, p, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xb.y, 1, (int)wbv.y, p, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xb.z, 1, (int)wbv.z, p, 0);
            p = __builtin_amdgcn_sudot4(1, (int)xb.w, 1, (int)wbv.w, p, 0);
            float c = (float)p * SG[(int64_t)m * (K >> 5) + g]
                            * AS[(int64_t)m * as_pitch + (g >> 2)] * sc * 0.5f;
            acc[m] += c;                      /* per-lane containment: NO fold */
        }
    }
    /* single tail reduce: full-wave butterfly at row end (donor's transposed tail) */
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        float v = acc[m];
        if (dpp) {
            v += __uint_as_float(__builtin_amdgcn_update_dpp(
                __float_as_uint(v), __float_as_uint(v), 0x161, 0xf, 0xf, 0));
            v += __uint_as_float(__builtin_amdgcn_update_dpp(
                __float_as_uint(v), __float_as_uint(v), 0x162, 0xf, 0xf, 0));
            v += __uint_as_float(__builtin_amdgcn_update_dpp(
                __float_as_uint(v), __float_as_uint(v), 0x164, 0xf, 0xf, 0));
            v += __uint_as_float(__builtin_amdgcn_update_dpp(
                __float_as_uint(v), __float_as_uint(v), 0x168, 0xf, 0xf, 0));
            v += __uint_as_float(q8x_perm_x16(__float_as_uint(v)));
        } else {
            v += __shfl_xor(v, 1);
            v += __shfl_xor(v, 2);
            v += __shfl_xor(v, 4);
            v += __shfl_xor(v, 8);
            v += __shfl_xor(v, 16);
        }
        if (t == 0 && m < live) Y[(int64_t)m * y_pitch + r] = f32_to_bf16_m4(v);
    }
}

/* explicit instantiations, shipped TU parity (MT = 1,2,4,8) */
template __global__ void q44_formA_kernel<1>(const uint8_t*, const int8_t*, const float*,
    const float*, uint16_t*, int, int, int, int64_t, int64_t, int, int, int);
template __global__ void q44_formA_kernel<2>(const uint8_t*, const int8_t*, const float*,
    const float*, uint16_t*, int, int, int, int64_t, int64_t, int, int, int);
template __global__ void q44_formA_kernel<4>(const uint8_t*, const int8_t*, const float*,
    const float*, uint16_t*, int, int, int, int64_t, int64_t, int, int, int);
template __global__ void q44_formA_kernel<8>(const uint8_t*, const int8_t*, const float*,
    const float*, uint16_t*, int, int, int, int64_t, int64_t, int, int, int);
template __global__ void q44_formB_kernel<1>(const uint8_t*, const int8_t*, const float*,
    const float*, uint16_t*, int, int, int, int64_t, int64_t, int, int, int);
template __global__ void q44_formB_kernel<2>(const uint8_t*, const int8_t*, const float*,
    const float*, uint16_t*, int, int, int, int64_t, int64_t, int, int, int);
template __global__ void q44_formB_kernel<4>(const uint8_t*, const int8_t*, const float*,
    const float*, uint16_t*, int, int, int, int64_t, int64_t, int, int, int);
template __global__ void q44_formB_kernel<8>(const uint8_t*, const int8_t*, const float*,
    const float*, uint16_t*, int, int, int, int64_t, int64_t, int, int, int);
