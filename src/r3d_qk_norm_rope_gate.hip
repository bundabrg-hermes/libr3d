/* r3d_qk_norm_rope_gate.hip -- libr3d's gated-attention QKV prologue in one pass, ported from
 * libr4d's r4d_qk_norm_rope_gate.hip (Apache-2.0, (c) radiance authors -- ported with
 * attribution; Q27, REVIEW_QUEUE window 39/40).
 *
 * Split [q|gate] per head, QK RMS norm over the head, partial NeoX RoPE from the shared cos/sin
 * table, and the gate copy -- and the `qk_norm_rope_gate_kv_store_bf16` form folds `kv_store`'s
 * cache write on, so a k head's block stores the rotated k and the v it already holds. The graph
 * template (arch/common/rad_block_attn_gated_fp8.h) probes the two op names through
 * rad_op_declared and wires them in place of rmsnorm(q) + rmsnorm(k) + rope(q) + rope(k)
 * (+ kv_store) whenever this library declares them; the unfused rows stay declared either way.
 *
 * Bit-identity anchors -- this is libr3d's OWN stack, so the anchors are libr3d's kernels, not
 * libr4d's (the donor's claims were against r4d's rmsnorm/rope; the reproductions below are the
 * same argument made element-for-element against the local references):
 *   * the norm reproduces r3d_rmsnorm's fold: one element a lane (head_dim <= 256 with 256
 *     threads), `r3d_wave_sum`'s shfl_down butterfly, the 8 wave partials summed in ascending
 *     order, `rsqrtf(sum/HD + eps)`, the single RTNE narrowing `r3d_f32_to_bf16(v*s*(g+wadd))`,
 *     and the value re-widened before the rotation -- exactly what rope reads back after rmsnorm
 *     stored it. Any other fold order breaks the bit-equality, hence 256 fixed threads.
 *   * the rotation consumes the cos/sin table `rope_table` already fills for the unfused path.
 *     r3d_rope.hip's rope_table comment states the contract this fusion relies on: the table and
 *     the in-place rope share the ladder width-for-width and the same sincosf, so reading the
 *     table agrees with `rope`'s own angles at the last operation of the ladder.
 *   * the bf16 cache write copies the same two bytes `kv_store_bf16` copies (identity, no
 *     rounding), at the layout its kernel walks: [n_blocks, kv_heads, block_size, 2*head_dim],
 *     K at [.., d], V at [.., head_dim + d], fully contiguous (checked at entry).
 *   * the fp8 cache fold of the donor is NOT ported: RDNA3 has no native E4M3 conversion path,
 *     the same scope rule as r3d_kvstore.hip ("the fp8 cache variants stay out of scope with the
 *     rest of fp8 -- RDNA3 has no native E4M3 path"), and the live gated stack runs a bf16 cache.
 *   * `wadd` is GemmaRMSNorm's `1 +` (Qwen3.5's QK gains are zero-centred), added to the gain in
 *     f32 after the load -- the identical expression r3d_rmsnorm computes.
 *
 * The k-side rotation is IN PLACE (w.k is both the `k` operand and the `k_out` operand, as the
 * template declares it): a lane's rotary partner is one of its sibling lanes' elements, so every
 * lane must read its input before any lane writes back. The donor's failure mode for a missing
 * barrier was a ~1% scatter of wrong elements that looks like rounding; the barrier before the
 * stores stays. Q31's WGP store law is not engaged: no barrier follows the kernel's global
 * stores (the KV-fold section adds none and never reads what other lanes stored).
 */

#include <hip/hip_runtime.h>
#include <cstring>
#include "r3d_dump.h"

#include "rad_abi.h"
#include "rad_plugin.h"

namespace {

constexpr uint32_t kThreads = 256;   /* fixed, not fitted: see the fold note above */
constexpr uint32_t kWave    = 32;
constexpr uint32_t kWaves   = kThreads / kWave;

/* Verbatim r3d_rope.hip's device-side narrowings (which mirror rad_plugin.h's host
 * conversions bit for bit), so the fused kernel rounds exactly where its references round. */
__device__ __forceinline__ float r3d_bf16_to_f32(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}
__device__ __forceinline__ uint16_t r3d_f32_to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* Verbatim r3d_rmsnorm.hip: the wave butterfly the reference folds with. */
__device__ __forceinline__ float r3d_wave_sum(float v) {
    const uint64_t full = ~0ull >> 32;
#pragma unroll
    for (uint32_t off = kWave / 2; off; off >>= 1)
        v += __shfl_down_sync(full, v, off);
    return v;
}

/* The paged cache the KV form writes, in one argument to stay inside the parameter budget.
 * The bf16 form needs no descales (kv_store_bf16's identity multiplies). */
struct QkKvStore {
    uint16_t*       cache;
    const uint16_t* v;
    const int32_t*  slot;
    int v_pitch, block_size, n_blocks;
};

/* The cache forms KV selects. fp8 is out of scope (file header). */
enum { QK_KV_NONE = 0, QK_KV_BF16 = 1 };

/* One block per (token, head) and one element per lane, required for bit-equality. Eight
 * elements per lane computes a different sum-of-squares fold (see the header). Lanes past the
 * head add exact zeros; lanes past the rotary range store their plain normed element. Indices
 * are clamped instead of guarding the loads -- a lane past the head reads its last element, a
 * lane past the rotary range reads its own element and table column 0 (which exists since
 * rot >= 2) -- and the values are used only where the guards allow. */
template <int HD, int KV = QK_KV_NONE>
__global__ __launch_bounds__(kThreads) void r3d_qk_norm_rope_gate_kernel(
    const uint16_t* __restrict__ QG, const uint16_t* __restrict__ K,
    uint16_t* __restrict__ QO, uint16_t* __restrict__ KO,
    uint16_t* __restrict__ GO,
    const void* __restrict__ WQ, const void* __restrict__ WK,
    const void* __restrict__ CS, const void* __restrict__ POS, long cs_rows,
    int nq, int nkv, int rot, float eps, float wadd,
    int qg_pitch, int k_pitch, int cs_f32, int pos_i64, int w_f32, QkKvStore kvs) {
    __shared__ float lds[kWaves];
    const int tok = blockIdx.x, h = blockIdx.y;
    const int t = threadIdx.x;
    const long p = pos_i64 ? ((const long*)POS)[tok] : (long)((const int*)POS)[tok];
    const int half = rot >> 1;
    const bool isk = h >= nq;
    const int lh = isk ? h - nq : h;
    const uint16_t* in = isk ? (K + (size_t)tok * k_pitch + (size_t)lh * HD)
                             : (QG + (size_t)tok * qg_pitch + (size_t)lh * 2 * HD);
    const void* w = isk ? WK : WQ;
    uint16_t* out = isk ? (KO + ((size_t)tok * nkv + lh) * HD)
                        : (QO + ((size_t)tok * nq + lh) * HD);
    const float* csf = (const float*)CS;
    const uint16_t* csb = (const uint16_t*)CS;

    /* Spelled once so this lane's element and its rotary partner cannot use different widths. */
    auto gain = [&](int i) -> float {
        return w_f32 ? ((const float*)w)[i] : r3d_bf16_to_f32(((const uint16_t*)w)[i]);
    };
    /* Rounded to bf16 before the rotation, then re-widened: what rope reads back after rmsnorm
     * stored it. Skipping the round trip would be more accurate and a different function. */
    auto normed = [&](float v, float g, float inv) -> float {
        return r3d_bf16_to_f32(r3d_f32_to_bf16(v * inv * (g + wadd)));
    };

    /* Everything the rotation reads is loaded before the reduction, so it is in flight during
     * the sum of squares: this lane's element and gain, its partner's, and its cos and sin. */
    const int tc = t < HD ? t : HD - 1;
    const bool rl = t < rot, first = t < half;
    const int pt = rl ? (first ? t + half : t - half) : tc;
    const int ci = rl ? (first ? t : t - half) : 0;
    /* The table row is clamped: a padded (negative) row or a position past the table's height
     * has no row rope_table wrote, and must not read outside the plane. */
    const size_t co = (size_t)(p < 0 ? 0 : (p < cs_rows ? p : cs_rows - 1)) * rot;
    const float vt = r3d_bf16_to_f32(in[tc]), gt = gain(tc);
    const float vp = r3d_bf16_to_f32(in[pt]), gp = gain(pt);
    const float c = cs_f32 ? csf[co + ci] : r3d_bf16_to_f32(csb[co + ci]);
    const float s = cs_f32 ? csf[co + half + ci] : r3d_bf16_to_f32(csb[co + half + ci]);

    float ss = 0.0f;
    if (t < HD) ss = vt * vt;
    /* r3d_rmsnorm's block_sum, transcribed: the butterfly, then the wave partials summed in
     * ascending order. The barrier orders only the LDS partials; the loads above stayed in
     * flight across it. */
    ss = r3d_wave_sum(ss);
    if ((t & 31) == 0) lds[t >> 5] = ss;
    __syncthreads();
    float sum = 0.0f;
#pragma unroll
    for (int wv = 0; wv < (int)kWaves; ++wv) sum += lds[wv];
    const float inv = rsqrtf(sum / (float)HD + eps);

    float res = 0.0f;
    if (t < HD) {
        res = normed(vt, gt, inv);
        if (rl) {
            const float xp = normed(vp, gp, inv);
            /* The two products and the sum/difference are separate f32 steps
             * (-ffp-contract=off), matching r3d_rope_kernel's x0*c - x1*s / x0*s + x1*c. */
            res = first ? (res * c - xp * s) : (xp * s + res * c);
        }
    }
    /* Every lane must read its input before any lane writes back: the engine rotates k in
     * place and a lane's partner is a memory read, so without this barrier it may already be
     * overwritten (file header; the failure is a ~1% wrong-element scatter). */
    __syncthreads();
    if (t < HD) {
        out[t] = r3d_f32_to_bf16(res);
        /* `gate_out` is optional: the gated fp8 block reads the gate straight from the
         * interleaved projection (gate_quant_fp8), so the template passes none. */
        if (!isk && GO) GO[((size_t)tok * nq + lh) * HD + t] = in[HD + t];
    }
    /* The cache write kv_store_bf16 would make next, from the same bytes: the rotated k as just
     * computed and the head's v element as the projection left it. Layout [n_blocks, kv_heads,
     * block_size, 2*head_dim], k then v, fully contiguous (entry-checked); a negative or
     * out-of-range slot stores nothing, exactly as r3d_kvstore.hip does. */
    if constexpr (KV != QK_KV_NONE) {
        if (isk && t < HD) {
            const int32_t sl = kvs.slot[tok];
            if (sl >= 0 && (uint32_t)sl / (uint32_t)kvs.block_size < (uint32_t)kvs.n_blocks) {
                const uint32_t blk = (uint32_t)sl / (uint32_t)kvs.block_size;
                const uint32_t off = (uint32_t)sl % (uint32_t)kvs.block_size;
                const size_t at = ((size_t)blk * nkv * kvs.block_size + off) * 2 * HD +
                                  (size_t)lh * kvs.block_size * 2 * HD;
                uint16_t* cch = kvs.cache + at;
                cch[t]      = r3d_f32_to_bf16(res);
                cch[HD + t] = kvs.v[(size_t)tok * kvs.v_pitch + (size_t)lh * HD + t];
            }
        }
    }
}

/* ------------------------------------------------------------------ host side */

constexpr int QK_QGATE = 0, QK_K = 1, QK_COSSIN = 2, QK_POS = 3, QK_WQ = 4, QK_WK = 5,
              QK_QOUT = 6, QK_KOUT = 7, QK_GOUT = 8, QK_V = 9, QK_SLOT = 10, QK_CACHE = 11;

/* Full row-major contiguity. The kernels above address operands flat (one row pitch a run),
 * so an operand whose strides are not the contiguous product is refused, not walked. The
 * engine's bands allocate these planes contiguously (the templates' operand shapes come from
 * the graph's own buffers); the check is the cold-call guard. */
bool rad_contig(const RadTensor* t) {
    int64_t want = 1;
    for (int r = (int)t->rank - 1; r >= 0; --r) {
        if (t->stride[r] != want) return false;
        want *= t->shape[r];
    }
    return true;
}

/* The body of both forms; KVF is the kv_store fold and `kvs` its cache, already checked. The
 * k-only form (`n_head` 0, the QSA indexer's plain-heads shape) follows the donor: every head
 * takes the k path and the absent q operands (null) are never dereferenced. */
template <int KVF>
int qk_body(const RadArgs* a, hipStream_t stream, const QkKvStore& kvs) {
    const RadTensor* qg = rad_arg_in(a, QK_QGATE);
    const RadTensor* k  = rad_arg_in(a, QK_K);
    const RadTensor* cs = rad_arg_in(a, QK_COSSIN);
    const RadTensor* po = rad_arg_in(a, QK_POS);
    const RadTensor* wq = rad_arg_in(a, QK_WQ);
    const RadTensor* wk = rad_arg_in(a, QK_WK);
    const RadTensor* qo = rad_arg_in(a, QK_QOUT);
    const RadTensor* ko = rad_arg_in(a, QK_KOUT);
    const RadTensor* go = rad_arg_in(a, QK_GOUT);
    if (!k || !cs || !po || !wk || !ko) return RAD_E_INVAL;
    if (!rad_contig(cs) || !rad_contig(po) || !rad_contig(wk)) return RAD_E_STRIDE;
    if (!rad_contig(ko)) return RAD_E_STRIDE;
    if (go && !rad_contig(go)) return RAD_E_STRIDE;

    long long hd = 0, nq = 0, nkv = 0, rot = 0;
    rad_args_geti(a, "head_dim", &hd);
    rad_args_geti(a, "n_head", &nq);
    rad_args_geti(a, "n_head_kv", &nkv);
    rad_args_geti(a, "rot", &rot);
    if (nq < 0 || nkv < 1) return RAD_E_SHAPE;
    if (nq >= 1 && (!qg || !wq || !qo)) return RAD_E_INVAL;
    if (qg && (!rad_contig(wq) || !rad_contig(qo) || qg->rank != 2)) return RAD_E_STRIDE;
    if (k->rank != 2 || !rad_contig(k)) return RAD_E_STRIDE;

    const long long qg_pitch = qg ? qg->stride[0] : 0;
    const long long k_pitch  = k->stride[0];
    if (qg_pitch < 0 || k_pitch < 0) return RAD_E_STRIDE;

    const long long M = qg ? qg->shape[0] : k->shape[0];
    if (M < 1 || M > 0x7ffffffLL) return RAD_E_SHAPE;
    if (rad_tensor_numel(po) < M) return RAD_E_SHAPE;
    if (hd != 64 && hd != 128 && hd != 256) return RAD_E_SHAPE;   /* the compiled set */
    if (rot < 2 || (rot & 1) || rot > hd) return RAD_E_SHAPE;
    const long long cs_rows = rad_tensor_numel(cs) / rot;
    if (cs_rows < 1 || rad_tensor_numel(cs) % rot) return RAD_E_SHAPE;

    const float eps  = (float)rad_args_getf_or(a, "eps", 1e-6);
    const float wadd = (float)rad_args_getf_or(a, "wadd", 0.0);
    /* The widths come from the operands: a cos/sin table in f32 and positions in int64 are
     * what the tensors say they are. One gain-width flag serves both gains, so they must
     * agree (f32 in every architecture plugin here, bf16 in vLLM). */
    if (cs->dtype != RAD_F32 && cs->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (po->dtype != RAD_I32 && po->dtype != RAD_I64) return RAD_E_DTYPE;
    if (wq && wq->dtype != wk->dtype) return RAD_E_DTYPE;
    if (wk->dtype != RAD_F32 && wk->dtype != RAD_BF16) return RAD_E_DTYPE;
    const int cs_f32  = (cs->dtype == RAD_F32) ? 1 : 0;
    const int pos_i64 = (po->dtype == RAD_I64) ? 1 : 0;
    const int w_f32   = (wk->dtype == RAD_F32) ? 1 : 0;

    const int nh = (int)(nq + nkv);
#define R3D_QKG_L(H)                                                              \
    hipLaunchKernelGGL((r3d_qk_norm_rope_gate_kernel<H, KVF>),                    \
        dim3((unsigned)M, (unsigned)nh), dim3(kThreads), 0, stream,               \
        (const uint16_t*)(qg ? qg->data : nullptr), (const uint16_t*)k->data,     \
        (uint16_t*)(qo ? qo->data : nullptr), (uint16_t*)ko->data,                \
        (uint16_t*)(go ? go->data : nullptr),                                     \
        wq ? wq->data : nullptr, wk->data, cs->data, po->data, (long)cs_rows,     \
        (int)nq, (int)nkv, (int)rot, eps, wadd, (int)qg_pitch, (int)k_pitch,      \
        cs_f32, pos_i64, w_f32, kvs)
    switch (hd) {
        case 64:  R3D_QKG_L(64);  break;
        case 128: R3D_QKG_L(128); break;
        default:  R3D_QKG_L(256); break;
    }
#undef R3D_QKG_L
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

} /* anonymous namespace */

/* Launch entries, operands positional in the schema order of oQkNormRopeGate /
 * oQkNormRopeGateKv (the r3d_rows.cpp registration; names and order verbatim libr4d). */

/* The prologue alone: rmsnorm(q) + rmsnorm(k) + rope(q) + rope(k) in one launch. */
static int r3d_qk_norm_rope_gate_impl(const RadArgs* a, RadStream s) {
    return qk_body<QK_KV_NONE>(a, (hipStream_t)(uintptr_t)s, QkKvStore{});
}
extern "C" int r3d_rad_qk_norm_rope_gate(const RadArgs* a, RadStream s)
    R3D_DUMP_WRAP("qk_norm_rope_gate", r3d_qk_norm_rope_gate_impl)

/* The prologue with the bf16 cache store folded in: qk_norm_rope_gate's operands, then
 * kv_store's `v`, `slot_mapping` and `kv_cache` at kv_store's own shapes. The k kv_store would
 * read is the prologue's own output, so the tail adds only the three. */
static int r3d_qk_norm_rope_gate_kv_store_impl(const RadArgs* a, RadStream s) {
    const RadTensor* qg = rad_arg_in(a, QK_QGATE);
    const RadTensor* k  = rad_arg_in(a, QK_K);
    const RadTensor* v  = rad_arg_in(a, QK_V);
    const RadTensor* sm = rad_arg_in(a, QK_SLOT);
    const RadTensor* kv = rad_arg_in(a, QK_CACHE);
    if (!k || !v || !sm || !kv) return RAD_E_INVAL;
    /* bf16 cache only on RDNA3 (file header). The row's constraints pin kv_dtype already;
     * this is the cold-call re-check. */
    const char* kvd = rad_args_gets(a, "kv_dtype");
    if ((kvd && std::strcmp(kvd, "bf16") != 0) ||
        v->dtype != RAD_BF16 || kv->dtype != RAD_BF16 || sm->dtype != RAD_I32)
        return RAD_E_DTYPE;
    if (!rad_contig(sm) || !rad_contig(v) || !rad_contig(kv)) return RAD_E_STRIDE;

    long long hd = 0, nkv = 0, bs = 0;
    rad_args_geti(a, "head_dim", &hd);
    rad_args_geti(a, "n_head_kv", &nkv);
    rad_args_geti(a, "block_size", &bs);
    if (hd <= 0 || nkv <= 0 || bs <= 0) return RAD_E_SHAPE;
    if (kv->rank != 4 || kv->shape[1] != nkv || kv->shape[2] != bs || kv->shape[3] != 2 * hd)
        return RAD_E_SHAPE;
    const long long M = qg ? qg->shape[0] : k->shape[0];
    if (v->rank != 2 || v->shape[1] < nkv * hd || v->shape[0] < M ||
        rad_tensor_numel(sm) != M)
        return RAD_E_SHAPE;

    /* The donor validates the descales even for the bf16 fold (where they are unused, as
     * kv_store_bf16's are); the same non-positive guard keeps a confused caller honest. */
    if (!(rad_args_getf_or(a, "k_scale", 1.0) > 0.0) ||
        !(rad_args_getf_or(a, "v_scale", 1.0) > 0.0))
        return RAD_E_INVAL;

    QkKvStore kvs{};
    kvs.v          = (const uint16_t*)v->data;
    kvs.slot       = (const int32_t*)sm->data;
    kvs.cache      = (uint16_t*)kv->data;
    kvs.v_pitch    = (int)v->stride[0];
    kvs.block_size = (int)bs;
    kvs.n_blocks   = (int)kv->shape[0];
    return qk_body<QK_KV_BF16>(a, (hipStream_t)(uintptr_t)s, kvs);
}
extern "C" int r3d_rad_qk_norm_rope_gate_kv_store(const RadArgs* a, RadStream s)
    R3D_DUMP_WRAP("qk_norm_rope_gate_kv_store_bf16", r3d_qk_norm_rope_gate_kv_store_impl)
