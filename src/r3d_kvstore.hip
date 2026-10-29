/* r3d_kvstore.hip -- libr3d's `kv_store` on RDNA3: scatter k/v into the paged cache.

 * The addressing is libref's (ref_attn.cpp ref_kv_store and its KVView, Apache-2.0,
 * (c) radiance authors -- ported with attribution): cache[n_blocks, kv_heads, block_size,
 * 2*head_dim], K at [.., d] and V at [.., head_dim + d], every stride read from the tensor
 * rather than assumed contiguous. For a bf16 cache the k_scale/v_scale descale factors are
 * unused (ref's identity multiplies); the fp8 cache variants stay out of scope with the rest
 * of fp8 -- RDNA3 has no native E4M3 path.
 *
 * The one deliberate divergence from ref: ref validates every slot up front and returns
 * RAD_E_SHAPE for an out-of-range one, because a host fallback can inspect its inputs before
 * touching them. A device row cannot cheaply do that, and the engine guarantees slots it
 * issues, so an out-of-range slot (like a negative one) simply stores nothing -- matching
 * libr4d's kv_store_bf16 row text ("a negative or out-of-range slot stores nothing") and
 * keeping kbench's red-zone runs fault-free rather than error-returning.
 *
 * One workgroup per token, threads striding the head*dim elements: a pure bf16 copy with an
 * indirection through the slot map, so width over cache stores is all there is to win here.
 *
 * The fp8 (E4M3) form -- kv_store_fp8, Q62 -- is the store side of --kv-cache-dtype fp8 at
 * head_dim 256 (the 27B's gated attention). The arithmetic contract is ref_kv_store's line
 * for line: an fp8 cache stores value * (1/k_scale) -- the reader's descale, default 1.0 --
 * narrowed by the ABI's saturating RTNE encoder. RDNA3 has no hardware E4M3 conversion, so
 * the encoder is the ABI's rad_f32_to_fp8e4m3 transcribed verbatim to __device__ below;
 * that makes every stored byte bit-equal to what libref's stt() writes for the same input,
 * at every scale the entry accepts.
 */

#include <hip/hip_runtime.h>
#include <cstring>
#include "r3d_dump.h"

#include "rad_abi.h"
#include "rad_plugin.h"

namespace {

constexpr uint32_t kThreads = 256;

/* One block per token. All strides in elements, resolved on the host side of the launch
 * exactly as ref's kv_view does; the bf16 value is copied without rounding (identity).
 * VEC (Q8, REVIEW_QUEUE): when every face is 16B-aligned and contiguous in 8-element runs
 * (serve's canonical paged layout), the walk copies uint4 -- byte-identical copies, 1/8 the
 * per-element divisions and stores. Odd strides keep the historical scalar walk (VEC=0). */
template <int VEC>
__global__ __launch_bounds__(kThreads) void kv_store_kernel(
    const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
    const int32_t* __restrict__ sm, uint16_t* __restrict__ kv,
    uint32_t head_dim, uint32_t n_head_kv, int32_t block_size, int64_t n_blocks,
    int64_t k_rs, int64_t k_hs, int64_t v_rs, int64_t v_hs,
    int64_t sb, int64_t sh, int64_t so, int64_t sd) {
    const int64_t t = blockIdx.x;
    const int32_t slot = sm[t];
    if (slot < 0) return;                       /* padding token: store nothing */
    const int64_t blk = slot / block_size, off = slot % block_size;
    if (blk >= n_blocks) return;                /* out of range: likewise (see header) */

    const uint16_t* __restrict__ kt = k + t * k_rs;
    const uint16_t* __restrict__ vt = v + t * v_rs;
    uint16_t* __restrict__ row = kv + blk * sb + off * so;

    const uint32_t per_head = head_dim;         /* elements each of k and v per head */
    if (VEC) {
        const uint32_t per_head8 = per_head >> 3;
        for (uint32_t L = threadIdx.x; L < n_head_kv * per_head8; L += kThreads) {
            const uint32_t h = L / per_head8, c8 = (L - h * per_head8) << 3;
            uint16_t* __restrict__ dst = row + (int64_t)h * sh + c8;
            *reinterpret_cast<uint4*>(dst) =
                *reinterpret_cast<const uint4*>(kt + (int64_t)h * k_hs + c8);
            *reinterpret_cast<uint4*>(dst + (int64_t)head_dim * sd) =
                *reinterpret_cast<const uint4*>(vt + (int64_t)h * v_hs + c8);
        }
    } else
    for (uint32_t e = threadIdx.x; e < n_head_kv * per_head; e += kThreads) {
        const uint32_t h = e / per_head, d = e % per_head;
        uint16_t* __restrict__ dst = row + (int64_t)h * sh + (int64_t)d * sd;
        dst[0]                 = kt[(int64_t)h * k_hs + d];   /* k contiguous within a head */
        dst[(int64_t)head_dim * sd] = vt[(int64_t)h * v_hs + d];
    }
}

/* Verbatim device transcription of the ABI's rad_f32_to_fp8e4m3 (abi/rad_plugin.h --
 * host-only static inline there, so the body is ported rather than called: the unannotated
 * header helpers are host under HIP). Saturates at 448 (never NaN/Inf for finite input,
 * the whole project's fp8 cast law), RTNE incl. the subnormal band, single NaN code kept.
 * Bit-equal to libref's stt(RAD_F8E4M3) by construction: the oracle stores through this
 * very function. */
__device__ __forceinline__ uint8_t r3d_f32_to_e4m3(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    {
        uint8_t  s = (uint8_t)((u >> 24) & 0x80u);
        uint32_t a = u & 0x7fffffffu;
        int e;
        uint32_t m;
        if (a > 0x7f800000u) return (uint8_t)(s | 0x7fu);     /* nan */
        if (a >= 0x43e00000u) return (uint8_t)(s | 0x7eu);    /* >= 448: clamp to the largest */
        e = (int)(a >> 23) - 127;
        m = a & 0x7fffffu;
        if (e < -10) return s;
        if (e < -6) {                                         /* subnormal fp8 */
            int sh = -6 - e + 20;
            uint32_t lo, q, half;
            m |= 0x800000u;
            lo   = m & ((1u << sh) - 1u);
            q    = m >> sh;
            half = 1u << (sh - 1);
            if (lo > half || (lo == half && (q & 1u))) ++q;
            return (uint8_t)(s | q);
        }
        {
            uint32_t q  = m >> 20;
            uint32_t lo = m & 0xfffffu;
            int ee;
            if (lo > 0x80000u || (lo == 0x80000u && (q & 1u))) ++q;
            ee = e + 7;
            if (q == 0x8u) { q = 0; ++ee; }
            if (ee > 15 || (ee == 15 && q >= 7u)) return (uint8_t)(s | 0x7eu);
            return (uint8_t)(s | ((uint32_t)ee << 3) | q);
        }
    }
}

__device__ __forceinline__ float r3d_bf16_f32(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);   /* the same widening ref's ld_dt does */
}

/* One bf16 -> (1/scale-scaled) -> E4M3 byte, the exact expression the scalar walk
 * performs; named so the 8-byte VEC packing below cannot be read wrong. */
__device__ __forceinline__ uint32_t r3d_e4m3_byte(uint16_t bits, float mul) {
    const uint8_t c = r3d_f32_to_e4m3(r3d_bf16_f32(bits) * mul);
    return (uint32_t)c;
}
/* Four bf16 (two packed f32 words) -> four E4M3 bytes in one uint32, LSB-first --
 * the plane's byte lane order. */
__device__ __forceinline__ uint32_t r3d_e4m3_pack4(uint32_t w0, uint32_t w1, float mul) {
    const uint32_t b0 = r3d_e4m3_byte((uint16_t)(w0      ), mul);
    const uint32_t b1 = r3d_e4m3_byte((uint16_t)(w0 >> 16), mul);
    const uint32_t b2 = r3d_e4m3_byte((uint16_t)(w1      ), mul);
    const uint32_t b3 = r3d_e4m3_byte((uint16_t)(w1 >> 16), mul);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}




/* E4M3 cache store (Q62). Same slot map and stride walk as the bf16 kernel above -- the
 * cache strides are ELEMENTS, which for fp8 are bytes; every address below re-derives
 * them from the launch scalars, so a wider-than-2*head_dim allocation walks correctly.
 * The fold is ref's: value * mul (mul = 1/k_scale, f32; identity 1.0 is exact) then the
 * saturating RTNE encode above. Multiplication precedes rounding: one narrowing, at the
 * encode, exactly where the oracle narrows.
 * VEC: the same alignment-gate shape as the bf16 row's, split per plane -- bf16 sources
 * load uint4 (8 elements, 16 B), byte destinations store uint2 (8 codes, 8 B). Encoding
 * is elementwise, so VEC and the scalar walk are byte-identical by construction; odd
 * strides keep the scalar path. */
template <int VEC>
__global__ __launch_bounds__(kThreads) void kv_store_f8_kernel(
    const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
    const int32_t* __restrict__ sm, uint8_t* __restrict__ kv,
    uint32_t head_dim, uint32_t n_head_kv, int32_t block_size, int64_t n_blocks,
    int64_t k_rs, int64_t k_hs, int64_t v_rs, int64_t v_hs,
    int64_t sb, int64_t sh, int64_t so, int64_t sd,
    float k_mul, float v_mul) {
    const int64_t t = blockIdx.x;
    const int32_t slot = sm[t];
    if (slot < 0) return;                       /* padding token: store nothing */
    const int64_t blk = slot / block_size, off = slot % block_size;
    if (blk >= n_blocks) return;                /* out of range: likewise (see header) */

    const uint16_t* __restrict__ kt = k + t * k_rs;
    const uint16_t* __restrict__ vt = v + t * v_rs;
    uint8_t* __restrict__ row = kv + blk * sb + off * so;

    const uint32_t per_head = head_dim;         /* elements each of k and v per head */
    if (VEC) {
        const uint32_t per_head8 = per_head >> 3;
        for (uint32_t L = threadIdx.x; L < n_head_kv * per_head8; L += kThreads) {
            const uint32_t h = L / per_head8, c8 = (L - h * per_head8) << 3;
            uint8_t* __restrict__ dst = row + (int64_t)h * sh + c8;
            const uint4 kc = *reinterpret_cast<const uint4*>(kt + (int64_t)h * k_hs + c8);
            const uint4 vc = *reinterpret_cast<const uint4*>(vt + (int64_t)h * v_hs + c8);
            uint2 kd, vd;
            kd.x = r3d_e4m3_pack4(kc.x, kc.y, k_mul);
            kd.y = r3d_e4m3_pack4(kc.z, kc.w, k_mul);
            vd.x = r3d_e4m3_pack4(vc.x, vc.y, v_mul);
            vd.y = r3d_e4m3_pack4(vc.z, vc.w, v_mul);
            *reinterpret_cast<uint2*>(dst) = kd;
            *reinterpret_cast<uint2*>(dst + (int64_t)head_dim * sd) = vd;
        }
    } else
    for (uint32_t e = threadIdx.x; e < n_head_kv * per_head; e += kThreads) {
        const uint32_t h = e / per_head, d = e % per_head;
        uint8_t* __restrict__ dst = row + (int64_t)h * sh + (int64_t)d * sd;
        dst[0]                      = r3d_f32_to_e4m3(r3d_bf16_f32(kt[(int64_t)h * k_hs + d]) * k_mul);
        dst[(int64_t)head_dim * sd] = r3d_f32_to_e4m3(r3d_bf16_f32(vt[(int64_t)h * v_hs + d]) * v_mul);
    }
}

} /* anonymous namespace */

/* Launch entry, operands positional in libref's kv_store schema order: k, v, slot_mapping,
 * kv_cache (INOUT: read as a descriptor, written through data). */
static int r3d_kv_store_impl(const RadArgs*, RadStream);
extern "C" int r3d_rad_kv_store(const RadArgs* a, RadStream s)
    R3D_DUMP_WRAP("kv_store_bf16", r3d_kv_store_impl)
static int r3d_kv_store_impl(const RadArgs* a, RadStream s) {
    const RadTensor* k   = rad_arg_in(a, 0);
    const RadTensor* v   = rad_arg_in(a, 1);
    const RadTensor* sm  = rad_arg_in(a, 2);
    const RadTensor* kv  = rad_arg_in(a, 3);
    if (!k || !v || !sm || !kv) return RAD_E_INVAL;
    if (k->dtype != RAD_BF16 || v->dtype != RAD_BF16 || kv->dtype != RAD_BF16)
        return RAD_E_DTYPE;
    if (sm->dtype != RAD_I32) return RAD_E_DTYPE;

    /* Geometry: parameters first, tensor extents as ref's fallbacks. The row table pins
     * kv_dtype=bf16 and block_size>=1; this is the cold-call re-check. */
    long long head_dim = 0, block_size = 0, n_head_kv = 0;
    if (!rad_args_geti(a, "head_dim", &head_dim) || head_dim <= 0)
        head_dim = rad_tensor_lastdim(k);
    if (!rad_args_geti(a, "block_size", &block_size) || block_size <= 0) return RAD_E_SHAPE;
    if (!rad_args_geti(a, "n_head_kv", &n_head_kv) || n_head_kv <= 0)
        n_head_kv = kv->rank == 4 ? kv->shape[1] : 0;
    if (head_dim <= 0 || n_head_kv <= 0) return RAD_E_SHAPE;

    /* This row serves the canonical 4-D paged cache only (ref also walks a 3-D fallback
     * layout; the engine always allocates the 4-D one, and refusing is the safe half). */
    if (kv->rank != 4) return RAD_E_SHAPE;
    if (kv->shape[1] != n_head_kv || kv->shape[2] != block_size ||
        kv->shape[3] < 2 * head_dim) return RAD_E_SHAPE;

    const int64_t M = rad_tensor_numel(sm);      /* ref: M = numel(slot_mapping) */
    if (M <= 0 || rad_tensor_numel(k) < M * n_head_kv * head_dim ||
        rad_tensor_numel(v) < M * n_head_kv * head_dim) return RAD_E_SHAPE;

    /* Strides per ref: within a head both k and v must be contiguous (the kernel indexes d
     * directly); every coarser stride is read from the tensor. */
    const int64_t k_ds = k->rank >= 2 ? k->stride[k->rank - 1] : 1;
    const int64_t v_ds = v->rank >= 2 ? v->stride[v->rank - 1] : 1;
    if (k_ds != 1 || v_ds != 1) return RAD_E_STRIDE;
    const int64_t k_hs = k->rank >= 3 ? k->stride[k->rank - 2] : head_dim * k_ds;
    const int64_t v_hs = v->rank >= 3 ? v->stride[v->rank - 2] : head_dim * v_ds;
    const int64_t k_rs = k->rank >= 2 ? k->stride[0] : n_head_kv * head_dim;
    const int64_t v_rs = v->rank >= 2 ? v->stride[0] : n_head_kv * head_dim;
    const int64_t sb = kv->stride[0], sh = kv->stride[1], so = kv->stride[2], sd = kv->stride[3];
    if (M > 0x7ffffffLL) return RAD_E_SHAPE;

    /* VEC gate (Q8): 16B = 8 bf16 elements; every plane must start and run in units of
     * 8, and every face must be contiguous within a head (sd==1, ds==1 already pinned). */
    const bool vec = head_dim % 8 == 0 && sd == 1 &&
                     sb % 8 == 0 && so % 8 == 0 && sh % 8 == 0 &&
                     k_rs % 8 == 0 && v_rs % 8 == 0 && k_hs % 8 == 0 && v_hs % 8 == 0 &&
                     (((uintptr_t)k->data | (uintptr_t)v->data | (uintptr_t)kv->data) & 15) == 0;

    hipStream_t stream = (hipStream_t)(uintptr_t)s;
    #define R3D_KVS_LA(V)                                                             \
        hipLaunchKernelGGL((kv_store_kernel<V>), dim3((uint32_t)M), dim3(kThreads), 0, \
                           stream, (const uint16_t*)k->data, (const uint16_t*)v->data,\
                           (const int32_t*)sm->data, (uint16_t*)kv->data,             \
                           (uint32_t)head_dim, (uint32_t)n_head_kv, (int32_t)block_size,\
                           kv->shape[0], k_rs, k_hs, v_rs, v_hs, sb, sh, so, sd)
    if (vec) R3D_KVS_LA(1); else R3D_KVS_LA(0);
    #undef R3D_KVS_LA
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* ---- kv_store_fp8 (Q62): the same scatter into an E4M3 cache. Operands, schema and the
 * shape checks are kv_store's (the strides stay ELEMENTS -- bytes for this cache); the
 * store fold is ref_kv_store's: value * (1/scale), saturating RTNE. The scales are read
 * the way ref reads them (default 1.0) and validated positive; the engine runs the
 * identity 1.0 today, but the row must be honest for every scale a caller passes. ---- */
static int r3d_kv_store_f8_impl(const RadArgs*, RadStream);
extern "C" int r3d_rad_kv_store_fp8(const RadArgs* a, RadStream s)
    R3D_DUMP_WRAP("kv_store_fp8", r3d_kv_store_f8_impl)
static int r3d_kv_store_f8_impl(const RadArgs* a, RadStream s) {
    const RadTensor* k   = rad_arg_in(a, 0);
    const RadTensor* v   = rad_arg_in(a, 1);
    const RadTensor* sm  = rad_arg_in(a, 2);
    const RadTensor* kv  = rad_arg_in(a, 3);
    if (!k || !v || !sm || !kv) return RAD_E_INVAL;
    if (k->dtype != RAD_BF16 || v->dtype != RAD_BF16 || kv->dtype != RAD_F8E4M3)
        return RAD_E_DTYPE;
    if (sm->dtype != RAD_I32) return RAD_E_DTYPE;

    long long head_dim = 0, block_size = 0, n_head_kv = 0;
    if (!rad_args_geti(a, "head_dim", &head_dim) || head_dim <= 0)
        head_dim = rad_tensor_lastdim(k);
    if (!rad_args_geti(a, "block_size", &block_size) || block_size <= 0) return RAD_E_SHAPE;
    if (!rad_args_geti(a, "n_head_kv", &n_head_kv) || n_head_kv <= 0)
        n_head_kv = kv->rank == 4 ? kv->shape[1] : 0;
    if (head_dim <= 0 || n_head_kv <= 0) return RAD_E_SHAPE;

    if (kv->rank != 4) return RAD_E_SHAPE;
    if (kv->shape[1] != n_head_kv || kv->shape[2] != block_size ||
        kv->shape[3] < 2 * head_dim) return RAD_E_SHAPE;

    const int64_t M = rad_tensor_numel(sm);
    if (M <= 0 || rad_tensor_numel(k) < M * n_head_kv * head_dim ||
        rad_tensor_numel(v) < M * n_head_kv * head_dim) return RAD_E_SHAPE;

    const int64_t k_ds = k->rank >= 2 ? k->stride[k->rank - 1] : 1;
    const int64_t v_ds = v->rank >= 2 ? v->stride[v->rank - 1] : 1;
    if (k_ds != 1 || v_ds != 1) return RAD_E_STRIDE;
    const int64_t k_hs = k->rank >= 3 ? k->stride[k->rank - 2] : head_dim * k_ds;
    const int64_t v_hs = v->rank >= 3 ? v->stride[v->rank - 2] : head_dim * v_ds;
    const int64_t k_rs = k->rank >= 2 ? k->stride[0] : n_head_kv * head_dim;
    const int64_t v_rs = v->rank >= 2 ? v->stride[0] : n_head_kv * head_dim;
    const int64_t sb = kv->stride[0], sh = kv->stride[1], so = kv->stride[2], sd = kv->stride[3];
    if (M > 0x7ffffffLL) return RAD_E_SHAPE;

    const double ksc = rad_args_getf_or(a, "k_scale", 1.0);
    const double vsc = rad_args_getf_or(a, "v_scale", 1.0);
    if (!(ksc > 0.0) || !(vsc > 0.0)) return RAD_E_INVAL;
    const float k_mul = 1.0f / (float)ksc;    /* ref's ks: f32 reciprocal of the f32 scale */
    const float v_mul = 1.0f / (float)vsc;

    /* VEC gate split per plane: the bf16 sources walk uint4 (8 elements, 16 B bases); the
     * byte cache walks uint2 (8 codes, 8 B bases). A non-unit cache last stride or a
     * misaligned plane keeps the scalar walk (bit-identical, see the kernel). */
    const bool vec = head_dim % 8 == 0 && sd == 1 &&
                     sb % 8 == 0 && so % 8 == 0 && sh % 8 == 0 &&
                     k_rs % 8 == 0 && v_rs % 8 == 0 && k_hs % 8 == 0 && v_hs % 8 == 0 &&
                     ((uintptr_t)kv->data & 7) == 0 &&
                     (((uintptr_t)k->data | (uintptr_t)v->data) & 15) == 0;

    hipStream_t stream = (hipStream_t)(uintptr_t)s;
    #define R3D_KVS8_LA(V)                                                              \
        hipLaunchKernelGGL((kv_store_f8_kernel<V>), dim3((uint32_t)M), dim3(kThreads), 0, \
                           stream, (const uint16_t*)k->data, (const uint16_t*)v->data, \
                           (const int32_t*)sm->data, (uint8_t*)kv->data,               \
                           (uint32_t)head_dim, (uint32_t)n_head_kv, (int32_t)block_size,\
                           kv->shape[0], k_rs, k_hs, v_rs, v_hs, sb, sh, so, sd, k_mul, v_mul)
    if (vec) R3D_KVS8_LA(1); else R3D_KVS8_LA(0);
    #undef R3D_KVS8_LA
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
