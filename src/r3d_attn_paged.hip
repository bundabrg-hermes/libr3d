/* r3d_attn_paged.hip -- libr3d's `attn_paged` on RDNA3: paged causal / sliding-window
 * attention over a [num_blocks, kv_heads, block_size, 2*head_dim] cache, any query length.
 *
 * The arithmetic contract is libref's (ref_attn.cpp ref_attn_paged, Apache-2.0,
 * (c) radiance authors -- ported with attribution), chain for chain:
 *   * score(j) = (sum over d of q[d] * (k[d] * k_descale)) * scale, the dot accumulated in
 *     f32 over ascending d -- so the dot here is summed SERIALLY by one thread in that
 *     order, not tree-reduced: the oracle's chain is a chain;
 *   * the max is over visible scores, then w(j) = exp(score - max) with the oracle's
 *     (float)exp(double) rounding, and acc[d] += w * (v[d] * v_descale) in ASCENDING key
 *     order, out = acc * (1/sum) narrowed once at the store (RTNE);
 *   * no visible key -> the zero row, never NaN;
 *   * E4M3 cache codes decode through the abi's rad_fp8e4m3_to_f32 -- exact into f32, so
 *     the f32 products match the oracle's LUT decode digit for digit. RDNA3's missing fp8
 *     hardware is the MMA path; a load-time conversion needs none of it, and an fp8 cache
 *     halves attention's bandwidth.
 * The oracle keeps its scores between the passes; we recompute each dot in pass two. Same
 * deterministic chain, same values, no jcap-sized buffer -- ref's own comment says keeping
 * them changes no value, only the recomputation is ours.
 *
 * Shape: one workgroup per (query row, q head), head_dim threads, thread d owning element d
 * of q and of the accumulator. All the paged-cache bookkeeping (sequence, position, block
 * table, descales, the serial dot) runs on thread 0 through shared broadcast. That is
 * latency-bound and deliberately so -- v1 must be RIGHT at every recorded geometry; the
 * wave64/WMMA split-KV designs libr4d runs on RDNA4 are the speed roadmap, and kbench
 * --bench will say exactly how far off they are.
 *
 * Operand reading mirrors ref_attn_paged exactly: stride-folded row pitches for q and out
 * (rank-3 slices read correctly -- the trap that zeroed rope's later rows), kv_view's four
 * strides, block_table row/col strides with rank-1 fallbacks, cu_seqlens optional (ragged
 * serving is the ordinary case: a prefill chunk and decode rows in one launch), window and
 * causal masking rules, unmapped (negative) block entries skipped, and keys past the block
 * table's width never read.
 */

#include <hip/hip_runtime.h>

#include "rad_abi.h"
#include "rad_plugin.h"
#include "r3d_attn.h"

namespace {

using AttnParams = R3dAttnParams;   /* shared with the WMMA tile TU via r3d_attn.h */

/* ---- dtype-widening helpers (identical bit patterns to libref's ld_dt path) ---------- */

__device__ __forceinline__ float bf16_f32(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}
__device__ __forceinline__ uint16_t f32_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* OCP E4M3 (e4m3fn, bias 7, 0x7f/0xff NaN, no infinity), the KV cache format at
 * --kv-cache-dtype fp8. Device mirror of the abi's rad_fp8e4m3_to_f32 (which is host-only;
 * libr4d keeps the same kind of mirror in r4d_attn_paged_window.hip): the normal path
 * shifts the fields into f32 bits, the subnormal path multiplies by 2^-9 -- exact, since
 * every E4M3 value is a dyadic rational an f32 holds whole. Both forms agree on all 256
 * codes, and the oracle decodes through the abi's table of this very function. */
__device__ __forceinline__ float fp8e4m3_dev(unsigned b) {
    const unsigned e = (b >> 3) & 15u, m = b & 7u;
    float v = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0.001953125f;
    if (e == 15u && m == 7u) v = __uint_as_float(0x7fc00000u);
    return __uint_as_float(__float_as_uint(v) | ((b & 0x80u) << 24));
}

__device__ __forceinline__ float ld_int_dev(uint32_t dt, const void* p, int64_t i) {
    switch (dt) {
        case RAD_I32: return (int64_t)((const int32_t*)p)[i];
        case RAD_U32: return (int64_t)((const uint32_t*)p)[i];
        case RAD_I64: return ((const int64_t*)p)[i];
        default:      return 0;   /* other int dtypes refused at the entry point */
    }
}

__device__ __forceinline__ float ldt_dev(const RadTensor* t, int64_t off) {
    switch (t->dtype) {
        case RAD_F32:  return ((const float*)t->data)[off];
        case RAD_BF16: return bf16_f32(((const uint16_t*)t->data)[off]);
        default:       return 0.0f;  /* f16 descales refused at the entry point */
    }
}

/* ---- one launch's scalars; strides/pitches computed host-side exactly as the oracle
 *      computes them from the tensor headers, so the device loop is pure addressing.
 *      The struct itself lives in r3d_attn.h (the WMMA tile reads the same fields). ---- */

/* ref_attn.cpp key_at: key j's slot offset in the cache, or -1 when the mask hides it or
 * its block is unmapped. Same rules, same order of tests. */
__device__ __forceinline__ int64_t key_at(const AttnParams& P, int64_t s, int64_t pos,
                                          int64_t kh, int64_t j) {
    if (P.causal && j > pos) return -1;
    if (P.window > 0) {
        const int64_t d = pos - j;
        if (P.causal ? (d >= P.window) : (d >= P.window || d <= -P.window)) return -1;
    }
    const int64_t blk =
        ld_int_dev(P.bt_dt, P.bt, s * P.bt_rs + (j / P.block_size) * P.bt_cs);
    return blk < 0 ? -1
                   : blk * P.kv_sb + kh * P.kv_sh + (j % P.block_size) * P.kv_so;
}

/* One (query row, q head) per workgroup; thread d owns output element d.
 * QKV: q and kv dtypes are template parameters so the loads are branch-free; the store
 * dtype is a runtime branch at the single narrowing point. */
template <int QDT, int KVDT>
__global__ void attn_paged_kernel(AttnParams P) {
    extern __shared__ float p[];                 /* head_dim element products, one dot */
    __shared__ int64_t sh_kb;                    /* key j's cache offset, or -1 */
    __shared__ int64_t sh_s, sh_qlen, sh_i;      /* sequence, its q length, row index in it */
    __shared__ int64_t sh_ctx, sh_pos, sh_kh;
    __shared__ float sh_kds, sh_vds, sh_w, sh_mx;

    const int64_t d  = threadIdx.x;
    const int64_t h  = blockIdx.x % P.q_heads;
    const int64_t tok = blockIdx.x / P.q_heads;

    /* Thread 0 derives the row's sequence bookkeeping once; everyone else reads it back.
     * cu_seqlens absent, every sequence has the same query length (ref's rule). */
    if (d == 0) {
        int64_t s, qlen, i;
        if (P.cus) {
            int64_t x = 0;
            while (x + 1 < P.n_seq &&
                   ld_int_dev(P.cus_dt, P.cus, x + 1) <= tok) ++x;   /* ref's seq_of, linear */
            s = x;
            qlen = ld_int_dev(P.cus_dt, P.cus, s + 1) - ld_int_dev(P.cus_dt, P.cus, s);
            i = tok - ld_int_dev(P.cus_dt, P.cus, s);
        } else {
            s = tok / P.q_len_uniform;
            qlen = P.q_len_uniform;
            i = tok - s * P.q_len_uniform;
        }
        sh_s = s; sh_qlen = qlen; sh_i = i;
    }
    __syncthreads();
    if (sh_qlen <= 0) return;

    const int64_t s = sh_s;
    if (d == 0) {
        const int64_t ctx = ld_int_dev(P.su_dt, P.su, s);
        sh_ctx = ctx;
        sh_pos = ctx - sh_qlen + sh_i;
        sh_kh  = h / P.gqa;
        sh_kds = P.kdes ? ldt_dev(P.kdes, s * P.kv_heads + sh_kh) : 1.0f;
        sh_vds = P.vdes ? ldt_dev(P.vdes, s * P.kv_heads + sh_kh) : 1.0f;
    }
    __syncthreads();

    const int64_t qb = tok * P.q_rs + h * P.q_hs;
    const float qd = (QDT == RAD_BF16)
        ? bf16_f32(((const uint16_t*)P.q)[qb + d * P.q_ds])
        : ((const float*)P.q)[qb + d * P.q_ds];

    const int64_t kh = sh_kh, pos = sh_pos;
    const float kds = sh_kds, vds = sh_vds;
    const int64_t jend = min<int64_t>(sh_ctx, P.jcap);

    /* pass one: the max of the visible scores, dots in ascending d, keys in order. */
    float mx = -INFINITY;
    for (int64_t j = 0; j < jend; ++j) {
        if (d == 0) sh_kb = key_at(P, s, pos, kh, j);
        __syncthreads();
        const int64_t kb = sh_kb;
        if (kb >= 0) {
            const float k = (KVDT == RAD_BF16)   ? bf16_f32(((const uint16_t*)P.kv)[kb + d * P.kv_sd])
                            : (KVDT == RAD_F32)  ? ((const float*)P.kv)[kb + d * P.kv_sd]
                                                 : fp8e4m3_dev(((const uint8_t*)P.kv)[kb + d * P.kv_sd]);
            p[d] = qd * (k * kds);
        }
        __syncthreads();
        if (d == 0 && kb >= 0) {
            float dot = 0.0f;
            for (int64_t t = 0; t < P.head_dim; ++t) dot += p[t];   /* the oracle's chain */
            const float sc = dot * P.scale;
            if (sc > mx) mx = sc;
        }
    }

    const int64_t ob = tok * P.o_rs + h * P.o_hs;

    /* thread 0 alone saw the scores; every thread's zero-row test and pass-two exp read
     * THE SAME max. (First cut kept mx thread-local: threads 1..hd-1 stayed at -inf,
     * took the zero-row return, and stored head_dim-1 zero columns -- repro_attn names
     * this failure instantly: d=0 alive, d>0 zero.) */
    if (d == 0) sh_mx = mx;
    __syncthreads();
    mx = sh_mx;

    /* No visible key (an empty sliding window) gives a zero row, not NaN. */
    if (mx == -INFINITY) {
        if (P.out_f32) ((float*)P.out)[ob + d * P.o_ds] = 0.0f;
        else ((uint16_t*)P.out)[ob + d * P.o_ds] = 0;
        return;
    }

    /* pass two: exp and accumulate in ascending key order. The dot is recomputed through
     * the same chain as pass one -- same value, no score buffer. Each thread folds w into
     * its own copy of `sum` so the (identical) scalar sum needs no extra broadcast. */
    float acc = 0.0f, sum = 0.0f;
    for (int64_t j = 0; j < jend; ++j) {
        if (d == 0) sh_kb = key_at(P, s, pos, kh, j);
        __syncthreads();
        const int64_t kb = sh_kb;
        if (kb >= 0) {
            const float k = (KVDT == RAD_BF16)   ? bf16_f32(((const uint16_t*)P.kv)[kb + d * P.kv_sd])
                            : (KVDT == RAD_F32)  ? ((const float*)P.kv)[kb + d * P.kv_sd]
                                                 : fp8e4m3_dev(((const uint8_t*)P.kv)[kb + d * P.kv_sd]);
            p[d] = qd * (k * kds);
        }
        __syncthreads();
        if (d == 0 && kb >= 0) {
            float dot = 0.0f;
            for (int64_t t = 0; t < P.head_dim; ++t) dot += p[t];
            sh_w = (float)exp((double)(dot * P.scale - mx));   /* ref_exp's rounding */
        }
        __syncthreads();
        if (kb >= 0) {
            const float w = sh_w;
            sum += w;
            const int64_t vb = kb + P.head_dim * P.kv_sd;
            const float v = (KVDT == RAD_BF16)   ? bf16_f32(((const uint16_t*)P.kv)[vb + d * P.kv_sd])
                            : (KVDT == RAD_F32)  ? ((const float*)P.kv)[vb + d * P.kv_sd]
                                                 : fp8e4m3_dev(((const uint8_t*)P.kv)[vb + d * P.kv_sd]);
            acc += w * (v * vds);
        }
    }

    const float out = acc * (sum > 0.0f ? 1.0f / sum : 0.0f);
    if (P.out_f32) ((float*)P.out)[ob + d * P.o_ds] = out;
    else ((uint16_t*)P.out)[ob + d * P.o_ds] = f32_bf16(out);
}

/* ref_gemm.h / ref_common row_stride: fold trailing dims until they span `cols`. */
int64_t row_stride(const RadTensor* t, int64_t cols) {
    if (!t || t->rank < 2) return cols;
    int64_t span = 1;
    for (int i = (int)t->rank - 1; i >= 0; --i) {
        span *= t->shape[i];
        if (span == cols) return (i > 0) ? t->stride[i - 1] : cols;
        if (span > cols) break;
    }
    return t->stride[t->rank - 2];
}

bool int_dt_ok(uint32_t dt) {
    return dt == RAD_I32 || dt == RAD_U32 || dt == RAD_I64;
}

} /* anonymous namespace */

/* Launch entry; operands positional in the attn_paged schema:
 * q, kv_cache, block_table, seqused, [k_descale], [v_descale], [cu_seqlens], out. */
extern "C" int r3d_rad_attn_paged(const RadArgs* a, RadStream s) {
    if (a->n_t < 8) return RAD_E_INVAL;
    const RadTensor* q   = rad_arg_in(a, 0);
    const RadTensor* kv  = rad_arg_in(a, 1);
    const RadTensor* bt  = rad_arg_in(a, 2);
    const RadTensor* su  = rad_arg_in(a, 3);
    const RadTensor* o   = rad_arg_in(a, 7);
    const RadTensor* kdes = rad_arg_in(a, 4);   /* null when absent or data-less */
    const RadTensor* vdes = rad_arg_in(a, 5);
    const RadTensor* cus  = rad_arg_in(a, 6);
    if (!q || !kv || !bt || !su || !o) return RAD_E_INVAL;

    AttnParams P{};
    P.q = q->data; P.kv = kv->data; P.out = o->data;
    P.bt = bt->data; P.su = su->data; P.cus = cus ? cus->data : nullptr;
    P.kdes = kdes; P.vdes = vdes;

    /* The oracle's parameter reads with the oracle's defaults. */
    P.head_dim = rad_args_geti_or(a, "head_dim", q->rank >= 3 ? q->shape[q->rank - 1] : 0);
    P.gqa      = rad_args_geti_or(a, "gqa", 1);
    P.block_size = rad_args_geti_or(a, "block_size", kv->rank == 4 ? kv->shape[2] : 0);
    P.causal   = (int)rad_args_geti_or(a, "causal", 1);
    P.window   = rad_args_geti_or(a, "window", 0);
    if (P.head_dim <= 0 || P.gqa <= 0 || P.block_size <= 0) return RAD_E_SHAPE;
    if (P.head_dim > 1024) return RAD_E_UNSUPPORTED;   /* one thread per element, block cap */

    if (kv->rank != 4) return RAD_E_SHAPE;             /* ref's kv_heads derives from rank 4 */
    P.kv_heads = kv->shape[1];
    if (P.kv_heads <= 0) return RAD_E_SHAPE;
    if (kv->shape[2] != P.block_size) return RAD_E_SHAPE;
    if (kv->shape[3] < 2 * P.head_dim) return RAD_E_SHAPE;
    P.q_heads = P.gqa * P.kv_heads;
    if (q->rank >= 3 && q->shape[q->rank - 2] != P.q_heads) return RAD_E_SHAPE;

    P.n_seq = rad_tensor_numel(su);
    const int64_t rows = rad_tensor_numel(q) / (P.q_heads * P.head_dim);
    if (P.n_seq <= 0 || rows <= 0) return RAD_E_SHAPE;
    if (!cus && rows % P.n_seq != 0) return RAD_E_SHAPE;
    if (cus && rad_tensor_numel(cus) != P.n_seq + 1) return RAD_E_SHAPE;
    if (!cus) P.q_len_uniform = rows / P.n_seq;
    if (rad_tensor_numel(o) < rows * P.q_heads * P.head_dim) return RAD_E_SHAPE;

    /* Dtypes the port serves. The scale multiplies stay f32 regardless. */
    if (q->dtype != RAD_BF16 && q->dtype != RAD_F32) return RAD_E_DTYPE;
    if (kv->dtype != RAD_BF16 && kv->dtype != RAD_F32 && kv->dtype != RAD_F8E4M3)
        return RAD_E_DTYPE;
    if (o->dtype != RAD_BF16 && o->dtype != RAD_F32) return RAD_E_DTYPE;
    P.bt_dt = bt->dtype; P.su_dt = su->dtype;
    if (!int_dt_ok(P.bt_dt) || !int_dt_ok(P.su_dt)) return RAD_E_DTYPE;
    if (cus) { P.cus_dt = cus->dtype; if (!int_dt_ok(P.cus_dt)) return RAD_E_DTYPE; }
    /* su and cus are indexed linearly here; ref reads them through offlin, which is the
     * identity only for contiguous rank-1. A pitched one is refused, not approximated. */
    if (su->rank != 1 || su->stride[0] != 1) return RAD_E_STRIDE;
    if (cus && (cus->rank != 1 || cus->stride[0] != 1)) return RAD_E_STRIDE;
    if (kdes && kdes->dtype != RAD_F32 && kdes->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (vdes && vdes->dtype != RAD_F32 && vdes->dtype != RAD_BF16) return RAD_E_DTYPE;

    /* Pitches and views, host-side, exactly where the oracle derives them. */
    P.q_ds = q->rank >= 2 ? q->stride[q->rank - 1] : 1;
    P.q_hs = q->rank >= 3 ? q->stride[q->rank - 2] : P.head_dim * P.q_ds;
    P.q_rs = row_stride(q, P.q_heads * P.head_dim);
    P.o_ds = o->rank >= 2 ? o->stride[o->rank - 1] : 1;
    P.o_hs = o->rank >= 3 ? o->stride[o->rank - 2] : P.head_dim * P.o_ds;
    P.o_rs = row_stride(o, P.q_heads * P.head_dim);
    P.kv_sb = kv->stride[0]; P.kv_sh = kv->stride[1];
    P.kv_so = kv->stride[2]; P.kv_sd = kv->stride[3];
    P.max_blocks = bt->rank >= 2 ? bt->shape[bt->rank - 1] : rad_tensor_numel(bt) / P.n_seq;
    P.bt_rs = bt->rank >= 2 ? bt->stride[bt->rank - 2] : P.max_blocks;
    P.bt_cs = bt->rank >= 2 ? bt->stride[bt->rank - 1] : 1;
    P.jcap = P.max_blocks * P.block_size;
    P.out_f32 = (o->dtype == RAD_F32);
    P.qdt = q->dtype; P.kvdt = kv->dtype; P.rows = rows;
    /* The default is computed in float, as ref computes it; a given scale arrives as the
     * stored double and narrows once, as ref's p_f32 does. */
    P.scale = (float)rad_args_getf_or(a, "scale",
                                      (double)(1.0f / sqrtf((float)P.head_dim)));

    const int64_t total = rows * P.q_heads;
    if (total <= 0 || total > 0x7fffffffLL) return RAD_E_SHAPE;

    const size_t shmem = (size_t)P.head_dim * sizeof(float);
    hipStream_t stream = (hipStream_t)(uintptr_t)s;

    /* Offer the launch to the WMMA tile first (r3d_attn_paged_wmma.hip). It sees the
     * same P the v1 kernel would have, computed in the same lines; it refuses what it
     * does not serve (chips without a gfx11xx object, q-f32 / kv-f32, head_dim outside
     * {64,128}, R3D_NO_WMMA=1) and a refusal is ordinary traffic, not an error -- v1
     * below was never launched in that case and serves the op exactly as before. */
    if (r3d_attn_paged_wmma_launch(P, stream) == RAD_OK) return RAD_OK;

    dim3 grid((unsigned)total);
    dim3 block((unsigned)P.head_dim);
    #define R3D_ATTN(Q, KV) hipLaunchKernelGGL((attn_paged_kernel<Q, KV>), grid, block, \
                                               shmem, stream, P)
    const int qdt = (int)q->dtype, kvdt = (int)kv->dtype;
    if      (qdt == RAD_BF16 && kvdt == RAD_BF16)   R3D_ATTN(RAD_BF16, RAD_BF16);
    else if (qdt == RAD_BF16 && kvdt == RAD_F32)    R3D_ATTN(RAD_BF16, RAD_F32);
    else if (qdt == RAD_BF16 && kvdt == RAD_F8E4M3) R3D_ATTN(RAD_BF16, RAD_F8E4M3);
    else if (qdt == RAD_F32  && kvdt == RAD_BF16)   R3D_ATTN(RAD_F32, RAD_BF16);
    else if (qdt == RAD_F32  && kvdt == RAD_F32)    R3D_ATTN(RAD_F32, RAD_F32);
    else                                             R3D_ATTN(RAD_F32, RAD_F8E4M3);
    #undef R3D_ATTN
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
