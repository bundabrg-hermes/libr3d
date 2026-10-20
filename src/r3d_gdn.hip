/* r3d_gdn.hip -- libr3d's first steps into the Gated DeltaNet family, the op set 48 of
 * the qwen3.5-27B trunk's 64 layers run on. The family opener is the cheapest member:
 * `gdn_gated_rmsnorm`, the epilogue norm the recurrent update also folds into itself.
 * The rest of the family (conv_update, kkt_solve, chunk_scan, recurrent_update) joins
 * this file; the arithmetic stays libref's (ref_gdn.cpp, Apache-2.0, (c) radiance
 * authors -- ported with attribution).
 *
 * gdn_conv_update (second member): the decode-band depthwise convolution over the rolling
 * window -- one block per (sequence, channel). The arithmetic is a width-tap f32 chain in
 * the ref's ascending j order over exactly-loaded bf16/f32 values (products of two bf16 are
 * exact in f32, so the conv is BIT-exact by construction, the same doctrine as the gemm
 * rows); the silu rides the band's double-rounded sigmoid the sigmoid_act row proved
 * bit-identical. The window contract (ref_ops.h §10) is mirrored verbatim: the history
 * reads at num_accepted-1 into shared BEFORE any state write, the rewrite is the shifted
 * tail then this step's tokens, and ref's per-sequence window checks become per-sequence
 * skips -- a guard can only be evaluated from device data here, and a too-shallow cache is
 * a caller bug ref rejects op-wide; skipping leaves the state exactly as it was.
 *
 * gdn_gated_rmsnorm (first member): o = x * rsqrt(mean(x^2) + eps) * w * act(z), act in
 * {silu, sigmoid}, one row per (token, value head):
 *   * the row sum is a block reduction in f32; ref accumulates it sequentially, and the
 *     tree-vs-sequence delta lives at 1e-7 on the row scale -- the same rounding class
 *     every reduced device row of this library already trades (r3d_rmsnorm);
 *   * the scaling keeps ref's exact op order: sc = 1/sqrt(ss/ch + eps), then the store
 *     chain ((x*sc)*w)*gate, each step a separate f32 (-ffp-contract=off) and the bf16
 *     narrowing happens once, at the store;
 *   * the gate reuses the band's double-rounded sigmoid (r3d_sigmoid), silu = z*sigmoid(z);
 *   * `z` may be a column slice of a wider fused row -- ref's gate_rowoff -- which the
 *     kernel walks as (row/heads)*z_pitch + (row%heads)*ch.
 */
#include <cstring>

#include <hip/hip_runtime.h>

#include "rad_abi.h"
#include "rad_plugin.h"

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kWarps   = kThreads / 32;

__device__ __forceinline__ float r3d_exp_f(float x) { return (float)exp((double)x); }
__device__ __forceinline__ float r3d_sigmoid(float x) { return 1.0f / (1.0f + r3d_exp_f(-x)); }

__device__ __forceinline__ float r3d_bf16_to_f32(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}
__device__ __forceinline__ uint16_t r3d_f32_to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float ldt2(const uint8_t* p, uint32_t dt, int64_t i) {
    return dt == RAD_F32 ? ((const float*)p)[i]
                         : r3d_bf16_to_f32(((const uint16_t*)p)[i]);
}
__device__ __forceinline__ void stt2(uint8_t* p, uint32_t dt, int64_t i, float v) {
    if (dt == RAD_F32) ((float*)p)[i] = v;
    else ((uint16_t*)p)[i] = r3d_f32_to_bf16(v);
}

/* Block sum of per-thread partials; order is tree by construction, see the file header
 * for why that is the same rounding class the shipped rmsnorm row already lives in. */
__device__ __forceinline__ float block_sum(float v, float* s_part) {
    for (uint32_t off = 16; off; off >>= 1)
        v += __shfl_down_sync(0xFFFFFFFFull, v, off);
    const uint32_t lane = threadIdx.x & 31u, wid = threadIdx.x >> 5;
    if (lane == 0) s_part[wid] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float r = s_part[0];
        for (uint32_t w = 1; w < kWarps; ++w) r += s_part[w];
        s_part[0] = r;
    }
    __syncthreads();
    return s_part[0];
}

/* One block per row. Pitches are in BYTES (rad_dtype_bytes applied at the entry); the
 * element strides within a row are 1 -- wider unit-strides are refused there, as every
 * other row of this library refuses them. z_heads > 1 selects the gate_rowoff slice
 * walk: z row = (r/z_heads)*z_tok + (r%z_heads)*ch elements. */
__global__ __launch_bounds__(kThreads) void gdn_gated_rmsnorm_kernel(
    const uint8_t* __restrict__ x, uint32_t dtX, int64_t x_pitch,
    const uint8_t* __restrict__ z, uint32_t dtZ, int64_t z_pitch,
    int64_t z_tok, int32_t z_heads,
    const uint8_t* __restrict__ w, uint32_t dtW,
    uint8_t* __restrict__ o, uint32_t dtO, int64_t o_pitch,
    int64_t ch, float eps, int sigm) {
    __shared__ float s_part[kWarps]; __shared__ float s_sc;
    const int64_t r = blockIdx.x;
    const uint8_t* __restrict__ xb = x + r * x_pitch;
    uint8_t* __restrict__ ob = o + r * o_pitch;
    const int64_t zb_elems = (z_heads > 1)
        ? (r / z_heads) * z_tok + (r % z_heads) * ch
        : r * (z_pitch / (dtZ == RAD_F32 ? 4 : 2));

    float ss = 0.0f;
    for (int64_t i = (int64_t)threadIdx.x; i < ch; i += (int64_t)kThreads) {
        const float u = ldt2(xb, dtX, i);
        ss += u * u;
    }
    ss = block_sum(ss, s_part);
    if (threadIdx.x == 0) s_sc = 1.0f / sqrtf(ss / (float)ch + eps);
    __syncthreads();
    const float sc = s_sc;

    const uint8_t* __restrict__ zb = z + zb_elems * (dtZ == RAD_F32 ? 4 : 2);
    for (int64_t i = (int64_t)threadIdx.x; i < ch; i += (int64_t)kThreads) {
        const float zz = ldt2(zb, dtZ, i);
        const float gate = sigm ? r3d_sigmoid(zz) : zz * r3d_sigmoid(zz);
        const float v = ((ldt2(xb, dtX, i) * sc) * ldt2(w, dtW, i)) * gate;
        stt2(ob, dtO, i, v);
    }
}

/* Ref's rowoff for the dense class this row serves: rank-2 [rows, cols] with a unit
 * last stride (pitch = stride[0]), or a bare rank-1 single row (pitch = cols). */
static inline int row2_pitch(const RadTensor* t, int64_t cols, int64_t* pitch) {
    if (!t || !t->data) return -1;
    if (t->rank == 1) {
        if (rad_tensor_numel(t) != cols || t->stride[0] != 1) return -2;
        *pitch = cols;
        return 0;
    }
    if (t->rank != 2 || t->shape[1] != cols || t->stride[1] != 1) return -3;
    *pitch = t->stride[0];
    return 0;
}

/* ---- gdn_conv_update ----------------------------------------------------------------------
 * One block per (sequence, channel): conv_dim blocks per sequence, exactly ref's
 * REF_PAR_FOR(N * conv_dim) decomposition. Shared holds the three tiny broadcasts ref
 * keeps per-channel in a stack frame: the history window, the weight taps, the bias.
 * Block size 64: decode blocks are launch-floored, and a wide block would only idle
 * more threads at slen == 1. */

constexpr uint32_t kConvThreads = 64;
constexpr int32_t  kConvHistMax = 16;   /* ref's REF_CONV_HIST_MAX */

/* Index-tensor reads (i32/u32/i64), same pattern as the attention rows' ldi_w. */
__device__ __forceinline__ int64_t ldi_g(uint32_t dt, const void* p, int64_t i) {
    switch (dt) {
        case RAD_I32: return (int64_t)((const int32_t*)p)[i];
        case RAD_U32: return (int64_t)((const uint32_t*)p)[i];
        case RAD_I64: return ((const int64_t*)p)[i];
        default:      return 0;
    }
}

__global__ __launch_bounds__(kConvThreads) void gdn_conv_update_kernel(
    const uint8_t* __restrict__ x, uint32_t dtX, int64_t x_pb,
    const uint8_t* __restrict__ w, uint32_t dtW, int64_t w_pb,
    const uint8_t* __restrict__ b, uint32_t dtB,
    uint8_t* __restrict__ cs, uint32_t dtCS, int64_t cs_s0b, int64_t cs_s1b,
    int64_t cs_slots, int64_t state_len,
    const void* __restrict__ sidx, uint32_t dtS,
    const void* __restrict__ nacc, uint32_t dtA,
    const void* __restrict__ cu, uint32_t dtC,
    uint8_t* __restrict__ q, uint32_t dtQ, int64_t q_s0b, int64_t q_s1b,
    uint8_t* __restrict__ k, uint32_t dtK, int64_t k_s0b, int64_t k_s1b,
    uint8_t* __restrict__ v, uint32_t dtV, int64_t v_s0b, int64_t v_s1b,
    int64_t HgK, int64_t Kd, int64_t K2, int64_t Vd,
    int32_t width) {
    const int64_t c = blockIdx.x, n = blockIdx.y;
    const int64_t bos  = ldi_g(dtC, cu, n);
    const int64_t slen = ldi_g(dtC, cu, n + 1) - bos;
    if (slen <= 0) return;
    const int64_t slot = sidx ? ldi_g(dtS, sidx, n) : n;
    if (slot < 0 || slot >= cs_slots) return;
    const int64_t off = nacc ? ldi_g(dtA, nacc, n) - 1 : 0;
    /* ref's window checks, scoped to a skip (see file header). */
    if (off < 0 || off + width - 1 > state_len) return;
    if (width > 1 && width - 2 + slen > state_len) return;

    uint8_t* __restrict__ csrow = cs + slot * cs_s0b + c * cs_s1b;
    const uint8_t* __restrict__ wrow = w + c * w_pb;

    __shared__ float s_hist[kConvHistMax + 1];
    __shared__ float s_w[kConvHistMax + 1];
    __shared__ float s_b;
    for (int32_t j = (int32_t)threadIdx.x; j < width - 1; j += (int32_t)kConvThreads)
        s_hist[j] = ldt2(csrow, dtCS, off + j);
    for (int32_t j = (int32_t)threadIdx.x; j < width; j += (int32_t)kConvThreads)
        s_w[j] = ldt2(wrow, dtW, j);
    if (threadIdx.x == 0) s_b = b ? ldt2(b, dtB, c) : 0.0f;
    __syncthreads();   /* also the barrier between the history read and the rewrite */

    /* The q/k/v split, verbatim ref conv_channel: channels are whole heads, so c picks
     * the tensor and the (head, dim) pair within it. */
    uint8_t* ob; uint32_t dtO; int64_t o_s0b, o_s1b, oh, od;
    if (c < HgK)            { ob = q; dtO = dtQ; o_s0b = q_s0b; o_s1b = q_s1b; oh = c / Kd;         od = c % Kd; }
    else if (c < K2)        { ob = k; dtO = dtK; o_s0b = k_s0b; o_s1b = k_s1b; oh = (c - HgK) / Kd; od = (c - HgK) % Kd; }
    else                    { ob = v; dtO = dtV; o_s0b = v_s0b; o_s1b = v_s1b; oh = (c - K2) / Vd;  od = (c - K2) % Vd; }
    uint8_t* __restrict__ obase = ob + oh * o_s1b;

    const int32_t w1 = width - 1;
    for (int64_t t = threadIdx.x; t < slen; t += kConvThreads) {
        float acc = s_b;
        for (int32_t j = 0; j < width; ++j) {
            const int64_t src = t - w1 + j;
            const float val = src >= 0 ? ldt2(x + (bos + src) * x_pb, dtX, c)
                                       : s_hist[src + w1];
            acc += val * s_w[j];
        }
        stt2(obase + (bos + t) * o_s0b, dtO, od, acc * r3d_sigmoid(acc));
    }

    /* Window rewrite, in place: the shifted tail, then this step's tokens at
     * t + width - 2. ref re-reads x for the commit (same value); we mirror it rather
     * than cache slen tokens in registers -- the read comes back from L1/L2. */
    for (int64_t i = threadIdx.x; i + 1 < width - 1; i += kConvThreads)
        stt2(csrow, dtCS, i, s_hist[i + 1]);
    for (int64_t t = threadIdx.x; t < slen; t += kConvThreads)
        stt2(csrow, dtCS, t + width - 2, ldt2(x + (bos + t) * x_pb, dtX, c));
}

} /* anonymous namespace */

extern "C" int r3d_rad_gdn_gated_rmsnorm(const RadArgs* a, RadStream s) {
    const RadTensor* x = rad_arg_in(a, 0);
    const RadTensor* z = rad_arg_in(a, 1);
    const RadTensor* w = rad_arg_in(a, 2);
    const RadTensor* o = rad_arg_in(a, 3);
    if (!x || !z || !w || !o) return RAD_E_INVAL;
    const bool dt_ok = x->dtype == RAD_F32 || x->dtype == RAD_BF16;
    const bool dt_z = z->dtype == RAD_F32 || z->dtype == RAD_BF16;
    const bool dt_w = w->dtype == RAD_F32 || w->dtype == RAD_BF16;
    const bool dt_o = o->dtype == RAD_F32 || o->dtype == RAD_BF16;
    if (!dt_ok || !dt_z || !dt_w || !dt_o) return RAD_E_DTYPE;

    long long ch_ll = 0;
    if (!rad_args_geti(a, "channels", &ch_ll))
        ch_ll = w->rank ? w->shape[w->rank - 1] : 0;
    const int64_t ch = ch_ll;
    if (ch <= 0 || ch > 0xffff) return RAD_E_SHAPE;

    const double eps_d = rad_args_getf_or(a, "eps", 1e-6);
    const float eps = (float)eps_d;

    const char* act = rad_args_gets(a, "act");
    const int sigm = act && !std::strcmp(act, "sigmoid");
    if (act && !sigm && std::strcmp(act, "silu")) return RAD_E_UNSUPPORTED;

    const int64_t rows = rad_tensor_numel(x) / ch;
    if (rows <= 0 || rows > 0xffff) return RAD_E_SHAPE;
    if (rad_tensor_numel(o) < rows * ch || rad_tensor_numel(z) < rows * ch) return RAD_E_SHAPE;
    if (rad_tensor_numel(w) < ch) return RAD_E_SHAPE;
    if (w->rank != 1 || w->stride[0] != 1) return RAD_E_STRIDE;

    int64_t xp, op;
    if (row2_pitch(x, ch, &xp)) return RAD_E_STRIDE;
    if (row2_pitch(o, ch, &op)) return RAD_E_STRIDE;

    /* The gate: a plain [rows, ch] row, or ref's gate_rowoff slice -- rank-2 whose row
     * is heads*ch wide, read heads-at-a-time. A rank-1 z is the single-row case. */
    int64_t z_pitch = 0, z_tok = 0; int32_t z_heads = 1;
    if (z->rank == 2 && z->shape[1] != ch && z->shape[1] % ch == 0 && z->stride[1] == 1) {
        z_heads = (int32_t)(z->shape[1] / ch);
        z_tok = z->stride[0];
    } else if (row2_pitch(z, ch, &z_pitch)) {
        return RAD_E_STRIDE;
    }

    gdn_gated_rmsnorm_kernel<<<dim3((uint32_t)rows), dim3(kThreads), 0,
                               (hipStream_t)(uintptr_t)s>>>(
        (const uint8_t*)x->data, x->dtype, rad_dtype_bytes(x->dtype, xp),
        (const uint8_t*)z->data, z->dtype, rad_dtype_bytes(z->dtype, z_pitch),
        z_tok, z_heads,
        (const uint8_t*)w->data, w->dtype,
        (uint8_t*)o->data, o->dtype, rad_dtype_bytes(o->dtype, op),
        ch, eps, sigm);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* Tensor fully dense: every stride is the product of the trailing extents (offlin's
 * identity condition, verbatim). Index tensors must be dense to be flat-read. */
static inline int idx_dense(const RadTensor* t) {
    int64_t acc = 1;
    for (int i = (int)t->rank - 1; i >= 0; --i) {
        if (t->shape[i] != 1 && t->stride[i] != acc) return 0;
        acc *= t->shape[i];
    }
    return 1;
}

static inline int idx_dt_ok(uint32_t dt) {
    return dt == RAD_I32 || dt == RAD_U32 || dt == RAD_I64;
}

extern "C" int r3d_rad_gdn_conv_update(const RadArgs* a, RadStream s) {
    const RadTensor* x   = rad_arg_in(a, 0);
    const RadTensor* w   = rad_arg_in(a, 1);
    const RadTensor* b   = rad_arg_in(a, 2);          /* optional */
    const RadTensor* cs  = rad_arg_in(a, 3);
    const RadTensor* sidx = rad_arg_in(a, 4);         /* optional */
    const RadTensor* nacc = rad_arg_in(a, 5);         /* required by schema, ref reads optional */
    const RadTensor* cu  = rad_arg_in(a, 6);
    const RadTensor* q   = rad_arg_in(a, 7);
    const RadTensor* k   = rad_arg_in(a, 8);
    const RadTensor* v   = rad_arg_in(a, 9);
    if (!x || !w || !cs || !cu || !q || !k || !v) return RAD_E_INVAL;

    const auto dt2_ok = [](uint32_t d) { return d == RAD_F32 || d == RAD_BF16; };
    if (!dt2_ok(x->dtype) || !dt2_ok(w->dtype) || !dt2_ok(cs->dtype) ||
        !dt2_ok(q->dtype) || !dt2_ok(k->dtype) || !dt2_ok(v->dtype) ||
        (b && !dt2_ok(b->dtype))) return RAD_E_DTYPE;
    if (sidx && (!idx_dt_ok(sidx->dtype) || !idx_dense(sidx))) return RAD_E_STRIDE;
    if (nacc && (!idx_dt_ok(nacc->dtype) || !idx_dense(nacc))) return RAD_E_STRIDE;
    if (!idx_dt_ok(cu->dtype) || !idx_dense(cu)) return RAD_E_STRIDE;

    long long K_ll = 0, V_ll = 0, W_ll = 0;
    if (!rad_args_geti(a, "head_k", &K_ll) || !rad_args_geti(a, "head_v", &V_ll) ||
        !rad_args_geti(a, "conv_width", &W_ll)) return RAD_E_SHAPE;
    const int64_t Kd = K_ll, Vd = V_ll, width = W_ll;
    if (Kd <= 0 || Vd <= 0 || width <= 0) return RAD_E_SHAPE;
    if (width - 1 > kConvHistMax) return RAD_E_UNSUPPORTED;   /* ref's own refusal */

    const int64_t N = rad_tensor_numel(cu) - 1;
    if (N <= 0 || N > 0xffff) return RAD_E_SHAPE;
    /* ref takes T from cu's last entry (device data); the row count x carries is the
     * same number for every valid call, and it is the only one a launcher can read. */
    const int64_t T = x->rank >= 2 ? x->shape[x->rank - 2] : 1;
    if (T <= 0) return RAD_E_SHAPE;
    const int64_t xrow = x->rank >= 2 ? x->stride[x->rank - 2] : 0;  /* 0 := tight below */
    if (x->rank >= 2 && x->stride[x->rank - 1] != 1) return RAD_E_STRIDE;

    const int64_t Hg = rad_tensor_numel(q) / (T * Kd);
    const int64_t H  = rad_tensor_numel(v) / (T * Vd);
    if (Hg <= 0 || H <= 0 || H % Hg != 0) return RAD_E_SHAPE;
    const int64_t conv_dim = 2 * Hg * Kd + H * Vd;
    const int64_t xrow_e = xrow ? xrow : conv_dim;
    if (rad_tensor_numel(x) < T * xrow_e) return RAD_E_SHAPE;
    if (rad_tensor_numel(w) < conv_dim * width) return RAD_E_SHAPE;
    if (sidx && rad_tensor_numel(sidx) < N) return RAD_E_SHAPE;
    if (nacc && rad_tensor_numel(nacc) < N) return RAD_E_SHAPE;

    /* w: [conv_dim, width] per ref's v2(w, width): tensor strides when rank >= 2. */
    const int64_t w_s0 = w->rank >= 2 ? w->stride[w->rank - 2] : width;
    const int64_t w_s1 = w->rank >= 2 ? w->stride[w->rank - 1] : 1;
    if (w_s1 != 1) return RAD_E_STRIDE;

    /* bias: ref reads element c * laststride; accept unit only (dense row). */
    if (b && (b->rank == 0 || b->stride[b->rank - 1] != 1)) return RAD_E_STRIDE;
    if (b && rad_tensor_numel(b) < conv_dim) return RAD_E_SHAPE;

    /* conv state: ref's conv_state_view -- [n_slots, conv_dim, state_len] with the
     * tensor's strides, or read as the tight layout when it arrives flat. */
    const int64_t state_len = cs->rank >= 3 ? cs->shape[cs->rank - 1] : (width > 1 ? width - 1 : 1);
    const int64_t cs_s0 = cs->rank >= 3 ? cs->stride[cs->rank - 3] : conv_dim * state_len;
    const int64_t cs_s1 = cs->rank >= 3 ? cs->stride[cs->rank - 2] : state_len;
    const int64_t cs_s2 = cs->rank >= 3 ? cs->stride[cs->rank - 1] : 1;
    if (cs_s2 != 1) return RAD_E_STRIDE;
    const int64_t cs_slots = cs->rank >= 3
        ? cs->shape[cs->rank - 3]
        : rad_tensor_numel(cs) / (conv_dim * state_len);
    if (cs_slots <= 0) return RAD_E_SHAPE;

    /* outputs: ref's v3(out, heads, dim) -- tensor strides at rank >= 3, tight below. */
    auto out_ok = [&](const RadTensor* t, int64_t heads, int64_t dim,
                      int64_t* s0, int64_t* s1) -> int {
        *s0 = t->rank >= 3 ? t->stride[t->rank - 3] : heads * dim;
        *s1 = t->rank >= 3 ? t->stride[t->rank - 2] : dim;
        const int64_t s2 = t->rank >= 3 ? t->stride[t->rank - 1] : 1;
        if (s2 != 1) return RAD_E_STRIDE;
        return rad_tensor_numel(t) < T * heads * dim ? RAD_E_SHAPE : RAD_OK;
    };
    int64_t q_s0, q_s1, k_s0, k_s1, v_s0, v_s1;
    int rc = out_ok(q, Hg, Kd, &q_s0, &q_s1); if (rc) return rc;
    rc = out_ok(k, Hg, Kd, &k_s0, &k_s1);     if (rc) return rc;
    rc = out_ok(v, H, Vd, &v_s0, &v_s1);      if (rc) return rc;

    const int32_t w32 = (int32_t)width;
    gdn_conv_update_kernel<<<dim3((uint32_t)conv_dim, (uint32_t)N), dim3(kConvThreads), 0,
                             (hipStream_t)(uintptr_t)s>>>(
        (const uint8_t*)x->data, x->dtype, rad_dtype_bytes(x->dtype, xrow_e),
        (const uint8_t*)w->data, w->dtype, rad_dtype_bytes(w->dtype, w_s0),
        b ? (const uint8_t*)b->data : nullptr, b ? b->dtype : 0,
        (uint8_t*)cs->data, cs->dtype,
        rad_dtype_bytes(cs->dtype, cs_s0), rad_dtype_bytes(cs->dtype, cs_s1),
        cs_slots, state_len,
        sidx ? sidx->data : nullptr, sidx ? sidx->dtype : 0,
        nacc ? nacc->data : nullptr, nacc ? nacc->dtype : 0,
        cu->data, cu->dtype,
        (uint8_t*)q->data, q->dtype, rad_dtype_bytes(q->dtype, q_s0), rad_dtype_bytes(q->dtype, q_s1),
        (uint8_t*)k->data, k->dtype, rad_dtype_bytes(k->dtype, k_s0), rad_dtype_bytes(k->dtype, k_s1),
        (uint8_t*)v->data, v->dtype, rad_dtype_bytes(v->dtype, v_s0), rad_dtype_bytes(v->dtype, v_s1),
        Hg * Kd, Kd, 2 * Hg * Kd, Vd, w32);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
