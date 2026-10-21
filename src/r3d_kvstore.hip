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
 */

#include <hip/hip_runtime.h>
#include "r3d_dump.h"

#include "rad_abi.h"
#include "rad_plugin.h"

namespace {

constexpr uint32_t kThreads = 256;

/* One block per token. All strides in elements, resolved on the host side of the launch
 * exactly as ref's kv_view does; the bf16 value is copied without rounding (identity). */
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
    for (uint32_t e = threadIdx.x; e < n_head_kv * per_head; e += kThreads) {
        const uint32_t h = e / per_head, d = e % per_head;
        uint16_t* __restrict__ dst = row + (int64_t)h * sh + (int64_t)d * sd;
        dst[0]                 = kt[(int64_t)h * k_hs + d];   /* k contiguous within a head */
        dst[(int64_t)head_dim * sd] = vt[(int64_t)h * v_hs + d];
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

    hipStream_t stream = (hipStream_t)(uintptr_t)s;
    hipLaunchKernelGGL(kv_store_kernel, dim3((uint32_t)M), dim3(kThreads), 0, stream,
                       (const uint16_t*)k->data, (const uint16_t*)v->data,
                       (const int32_t*)sm->data, (uint16_t*)kv->data,
                       (uint32_t)head_dim, (uint32_t)n_head_kv, (int32_t)block_size,
                       kv->shape[0], k_rs, k_hs, v_rs, v_hs, sb, sh, so, sd);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
