/* r3d_attn_paged_wmma.hip -- WMMA flash tile for attn_paged on RDNA3 (gfx1100 class).

 * Why this TU exists: the two-probe prefill fit (docs/COVERAGE.md, 2026-10-20) puts ~87%
 * of prefill time in attention, and the v1 kernel in r3d_attn_paged.hip runs it at
 * ~133 GFLOP/s -- one workgroup per (row, head), three syncthreads per KEY, and the
 * score dot serialized on thread 0. The card's matrix units were never invited.

 * Shape: one workgroup per (64-query tile, q head, sequence); 128 threads = 4 wave32s,
 * wave w owning query rows [16w, 16w+16). Keys stream in 32-wide tiles (the XTX's KV
 * pages are 16 tokens; a key tile spans at most two block-table entries, gathered per
 * row through the same key_at addressing v1 uses -- unmapped pages mark their columns
 * bad, and bad columns are skipped, never zero-scored, matching ref's key-skip rule).

 * Two passes, not FA2's online rescale: pass one walks K tiles computing S = Q.K^T with
 * WMMA and reduces per-row running max m and sum l; pass two RECOMPUTES the same S tiles
 * (bit-identical: same fragments, same order), forms P = exp2(S*scale2 - m) in bf16, and
 * accumulates O += P.V with WMMA. The reward for 1.5x the QK^T flops is that no WMMA
 * accumulator is ever rescaled mid-loop and no fragment needs a layout conversion --
 * accumulators go LDS->global once, at the end. At the card's tensor-core rate the extra
 * flops are noise; v1's latency was the disease.

 * Arithmetic drift vs libref (adjudicated by kbench rel_l2 + the KLD gate, same contract
 * class as the gemm m64 tile): the WMMA fold order is opaque (reassociation, not rounding
 * class), scores are scaled by the folded scale*kds*log2e instead of kds-per-k and
 * exp-after-max, exp2f replaces (float)exp(double), P narrows to bf16 before the PV
 * product, and v_descale folds at the final multiply instead of per key element. Every
 * one of these is sub-bf16-store rounding; the KLD gate -- built for exactly this
 * question -- is the arbiter, with the v1 kernel as its base arm via R3D_NO_WMMA=1.

 * The exp2 domain: m and l live in S*scale2 with scale2 = scale*kds*log2(e), so
 * w = exp2(S*scale2 - m) == exp(score - max_score) up to the folded-rounding drift above.

 * Serve geometry this tile is built for (MiniCPM5-2B): head_dim 128, 16 q heads / 2 kv
 * heads (gqa 8), block_size 16, bf16 cache. head_dim 64 compiles and runs on the same
 * template; q-f32 or kv-f32 caches, head_dim outside {64,128}, and every refused shape
 * return RAD_E_UNSUPPORTED and v1 serves them, unchanged, forever. Decode rows are
 * eligible too: a qlen=1 sequence wastes 63/64 of a query tile's FMAs but reads the same
 * bytes v1 reads -- decode is bandwidth-bound, and the tensor cores make the waste free
 * (measured, not assumed: the serve benchmark is the gate).

 * Chip gating and the R3D_NO_WMMA kill switch follow r3d_gemm_bf16_m64.hip's lesson: the
 * kernel body compiles behind the gfx1100..gfx1103 gate (rocWMMA's config gate rejects
 * family-generic targets), the LAUNCHER lives outside it and decides by gcnArchName at
 * runtime, so a generic-target build carries a stub the launcher never dispatches.
 */

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rad_abi.h"
#include "r3d_attn.h"

namespace {

/* dtype-widening helpers: bit-identical to the v1 kernel's (same contracts, see
 * r3d_attn_paged.hip); duplicated rather than shared to keep the two TUs independent --
 * these eleven lines are the most-reviewed eleven in the project. */
__device__ __forceinline__ float bf16_f32_w(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}
__device__ __forceinline__ uint16_t f32_bf16_w(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float fp8e4m3_w(unsigned b) {
    const unsigned e = (b >> 3) & 15u, m = b & 7u;
    float v = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0.001953125f;
    if (e == 15u && m == 7u) v = __uint_as_float(0x7fc00000u);
    return __uint_as_float(__float_as_uint(v) | ((b & 0x80u) << 24));
}
/* E4M3 -> bf16 bits: every code is a dyadic rational bf16 holds whole (4 significant
 * bits, exponent range -9..+8), so the narrowing is a truncation of an exact value. */
__device__ __forceinline__ uint16_t fp8_to_bf16_w(unsigned b) {
    return (uint16_t)(__float_as_uint(fp8e4m3_w(b)) >> 16);
}
__device__ __forceinline__ float ldt_w(const RadTensor* t, int64_t off) {
    switch (t->dtype) {
        case RAD_F32:  return ((const float*)t->data)[off];
        case RAD_BF16: return bf16_f32_w(((const uint16_t*)t->data)[off]);
        default:       return 0.0f;
    }
}
__device__ __forceinline__ float ldi_w(uint32_t dt, const void* p, int64_t i) {
    switch (dt) {
        case RAD_I32: return (int64_t)((const int32_t*)p)[i];
        case RAD_U32: return (int64_t)((const uint32_t*)p)[i];
        case RAD_I64: return ((const int64_t*)p)[i];
        default:      return 0;
    }
}

constexpr int kBM = 64, kBN = 32;           /* query rows x keys per step */
constexpr int kThreads = 128;               /* v1 body: 4 wave32s, 16 query rows each */
constexpr int kT = 256;                     /* v2 body (body2.inc): 8 wave32s; outside the
                                             * chip gate so the launcher can size its block */
constexpr float kLog2e = 1.44269504088896340736f;

} /* anonymous namespace */

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)

#include <rocwmma/rocwmma.hpp>

namespace {

namespace roCW = rocwmma;
using bf16 = roCW::bfloat16_t;


#include "r3d_attn_wmma_body.inc"    /* attn_paged_wmma_kernel<D, KVDT, SPLITK>  -- v1 schedule */
#include "r3d_attn_wmma_body2.inc"   /* attn_paged_wmma_kernel2<D, KVDT, SPLITK> -- v2 schedule */

} /* anonymous namespace */

#else  /* generic + non-RDNA3: kernel exists as a no-op so the symbol links; the
        * launcher's gcnArchName gate below never dispatches it on such a chip. */

namespace {
template <int D, int KVDT, int SPLITK>
__global__ __launch_bounds__(kThreads) void attn_paged_wmma_kernel(R3dAttnParams) {}
template <int D, int KVDT, int SPLITK>
__global__ __launch_bounds__(kT) void attn_paged_wmma_kernel2(R3dAttnParams) {}
} /* anonymous namespace */

#endif

#ifdef R3D_DBG
namespace { __global__ void r3d_dbg_null(int* p) { if (p) *p = 1; } }
#endif

int r3d_attn_paged_wmma_launch(const R3dAttnParams& P, hipStream_t stream) {
#ifdef R3D_DBG
    static int dbg_once = 0;
    if (!dbg_once) {
        dbg_once = 1;
        hipFuncAttributes at{};
        hipError_t e1 = hipFuncGetAttributes(&at, (const void*)&attn_paged_wmma_kernel<128, RAD_BF16, 0>);
        printf("DBG attr bf16 err=%d shared=%zu regs=%d\n", (int)e1, at.sharedSizeBytes, (int)at.numRegs);
        e1 = hipFuncGetAttributes(&at, (const void*)&attn_paged_wmma_kernel<128, RAD_F8E4M3, 0>);
        printf("DBG attr f8  err=%d shared=%zu regs=%d\n", (int)e1, at.sharedSizeBytes, (int)at.numRegs);
        int* nullobj = nullptr;
        if (hipMalloc(&nullobj, 4) == hipSuccess) {
            r3d_dbg_null<<<1, 64>>>(nullobj);
            printf("DBG null launch err=%d %s\n", (int)hipGetLastError(), hipGetErrorString(hipGetLastError()));
            hipDeviceSynchronize();
        }
    }
    printf("DBG gate qdt=%d kvdt=%d hd=%d rows=%d n_seq=%d q_heads=%d | BF16=%d F8E4M3=%d "
           "allowed=%d\n",
           (int)P.qdt, (int)P.kvdt, (int)P.head_dim, (int)P.rows, (int)P.n_seq, (int)P.q_heads,
           (int)RAD_BF16, (int)RAD_F8E4M3, (int)({ static int a = -1;
               if (a < 0) { hipDeviceProp_t d; int dev = 0;
                   if (hipGetDevice(&dev) == hipSuccess && hipGetDeviceProperties(&d, dev) == hipSuccess)
                       a = (strstr(d.gcnArchName, "gfx1100") || strstr(d.gcnArchName, "gfx1101")) ? 1 : 0;
                   else a = 0;
                   if (getenv("R3D_NO_WMMA")) a = 0; }
               a; }));
#endif
    static int allowed = -1;
    if (allowed < 0) {
        hipDeviceProp_t d;
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&d, dev) != hipSuccess)
            allowed = 0;
        else
            allowed = (strstr(d.gcnArchName, "gfx1100") || strstr(d.gcnArchName, "gfx1101") ||
                       strstr(d.gcnArchName, "gfx1102") || strstr(d.gcnArchName, "gfx1103")) ? 1 : 0;
        if (getenv("R3D_NO_WMMA")) allowed = 0;      /* one switch for every WMMA row */
    }
    if (!allowed) return RAD_E_UNSUPPORTED;
    if (P.qdt != RAD_BF16) return RAD_E_UNSUPPORTED;
    if (P.head_dim != 64 && P.head_dim != 128) return RAD_E_UNSUPPORTED;

    /* v1 is the default: the v2 256-thread schedule measured 0.43-0.80x at deep ctx
     * (bit-identical hashes, same-card bench 2026-10-20) -- one 40.7 KB-LDS block per
     * CU means every barrier stalls the whole CU, and cross-block overlap that v1's
     * two smaller blocks provided is gone. R3D_ATTN_V2=1 runs it for postmortem;
     * same .so, same shapes, every dtype either way. */
    static int v2 = -1;
    if (v2 < 0) v2 = getenv("R3D_ATTN_V2") ? 1 : 0;

    const unsigned tiles = (unsigned)((P.rows + kBM - 1) / kBM);
    if (P.n_seq <= 0 || P.n_seq > 65535 || tiles == 0 || tiles > 65535) return RAD_E_UNSUPPORTED;
    dim3 grid((unsigned)P.q_heads, (unsigned)P.n_seq, tiles);
    dim3 block(v2 ? kT : kThreads);

    const int kvbf = (P.kvdt == RAD_BF16);
    if (!kvbf && P.kvdt != RAD_F8E4M3) return RAD_E_UNSUPPORTED;
    #define R3D_LAUNCH(K, D, KV) hipLaunchKernelGGL((K<D, KV, 0>), grid, block, 0, stream, P)
    const int d128 = (P.head_dim == 128);
    #define R3D_ARMS(K)                                                        \
        if      (d128 &&  kvbf) R3D_LAUNCH(K, 128, RAD_BF16);                  \
        else if (d128 && !kvbf) R3D_LAUNCH(K, 128, RAD_F8E4M3);                \
        else if (!d128 &&  kvbf) R3D_LAUNCH(K, 64, RAD_BF16);                  \
        else                     R3D_LAUNCH(K, 64, RAD_F8E4M3);
    if (v2) { R3D_ARMS(attn_paged_wmma_kernel2) }
    else    { R3D_ARMS(attn_paged_wmma_kernel)  }
    #undef R3D_ARMS
    #undef R3D_LAUNCH
    hipError_t le = hipGetLastError();
#ifdef R3D_DBG
    printf("DBG launched grid(%u,%u,%u) err=%d %s\n",
           grid.x, grid.y, grid.z, (int)le, hipGetErrorString(le));
#endif
    return le == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
