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
 *
 * gdn_kkt_solve (third member): the chunk triangular inverse -- see the section comment for
 * the full shape; the short version is that ref's forward substitution is a per-column f32
 * chain, a chain a thread-owns-column kernel can walk in ref's exact order, so this row is
 * bit-exact against the oracle without ever touching a WMMA fragment (libr4d's blocked-merge
 * sibling on gfx12 is not).
 *
 * gdn_chunk_scan (fourth member): the chunked delta-rule scan -- the serial-chunk recurrence
 * means no chunk grid axis, so one block per (sequence, v head) carries the transposed f32
 * state in one-column-per-thread register arrays and walks every ref chain (ascending d and j,
 * literal bracketings) unrolled; bit-exact by construction the same way, see the section.
 *
 * gdn_conv_prep (sixth member) and gdn_conv_recurrent_update (seventh, the pair fusion): the
 * last two members of the family, and the unlock for the 27B serve -- their host-only
 * resolution pinned every block's whole weight group to the mapped tier, where the widened
 * f32 norms are illegal (COVERAGE "27B serve placement chain"). Like the recurrent row, the
 * phantom "needs a three-plane layout hook" note dissolved at first probe: every weight the
 * two ops read (conv w/b, A_log, dt_bias, norm_w) is plain single-plane in this container,
 * read AS STORED -- libr4d's own conv rows declare no hook either. See the section comments.
 */
#include <cstring>

#include <hip/hip_runtime.h>

#include "rad_abi.h"
#include "rad_plugin.h"

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kWarps   = kThreads / 32;

__device__ __forceinline__ float r3d_exp_f(float x) { return (float)exp((double)x); }
__device__ __forceinline__ float r3d_log1p_f(float x) { return (float)log1p((double)x); }
__device__ __forceinline__ float r3d_sigmoid(float x) { return 1.0f / (1.0f + r3d_exp_f(-x)); }
/* ref_common.h act_softplus, two-sided form with the large-x cutoff, verbatim: BOTH branches
 * take log1p of exp(+/-x) -- the exp is the whole function, as a -inf gate exponent proved. */
__device__ __forceinline__ float r3d_softplus(float x, float thr) {
    if (x > thr) return x;
    return x > 0.0f ? x + r3d_log1p_f(r3d_exp_f(-x)) : r3d_log1p_f(r3d_exp_f(x));
}

__device__ __forceinline__ float r3d_bf16_to_f32(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}
__device__ __forceinline__ uint16_t r3d_f32_to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
/* e4m3, the same encoder as r3d_band.hip's (which mirrors the ABI's rad_f32_to_fp8e4m3 that
 * libref's st_dt uses): clamp to the largest finite, round to nearest even, subnormals exact. */
__device__ __forceinline__ uint8_t r3d_f32_to_e4m3(float f) {
    const uint32_t u = __float_as_uint(f);
    const uint8_t  s = (uint8_t)((u >> 24) & 0x80u);
    const uint32_t a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return (uint8_t)(s | 0x7fu);     /* nan */
    if (a >= 0x43e00000u) return (uint8_t)(s | 0x7eu);    /* >= 448: clamp to the largest */
    const int e = (int)(a >> 23) - 127;
    uint32_t m = a & 0x7fffffu;
    if (e < -10) return s;                                /* below half the smallest subnormal */
    if (e < -6) {                                         /* subnormal fp8 */
        const int sh = -6 - e + 20;
        m |= 0x800000u;
        const uint32_t lo = m & ((1u << sh) - 1u);
        uint32_t q = m >> sh;
        const uint32_t half = 1u << (sh - 1);
        if (lo > half || (lo == half && (q & 1u))) ++q;
        return (uint8_t)(s | q);
    }
    uint32_t q = m >> 20;
    const uint32_t lo = m & 0xfffffu;
    int ee;
    if (lo > 0x80000u || (lo == 0x80000u && (q & 1u))) ++q;
    ee = e + 7;
    if (q == 0x8u) { q = 0; ++ee; }
    if (ee > 15 || (ee == 15 && q >= 7u)) return (uint8_t)(s | 0x7eu);
    return (uint8_t)(s | ((uint32_t)ee << 3) | q);
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
/* Ref's general row view (ref_common.h rowoff): a tensor read as [rows, ch] where ch is the
 * last dimension and a row r walks the leading dimensions with THEIR strides, so [T,H,V] is
 * correct even when H's stride is not V. nr == 0 means the linear fallback (row pitch in
 * elements); nr == -1 is ref's gate_rowoff heads slice. Passed by value to the kernel, which
 * repeats the identical walk per row. */
struct R3dRow {
    int32_t nr;                 /* leading dims to walk (0: linear, -1: gate heads slice) */
    int32_t shp[3];
    int64_t str[3];
    int64_t pitch;              /* nr == 0: row pitch, elements. nr == -1: z_tok */
    int32_t heads;              /* nr == -1 */
};

__device__ __forceinline__ int64_t r3d_rowbase(const R3dRow& v, int64_t r) {
    if (v.nr <= 0) {
        if (v.nr == 0) return r * v.pitch;
        return (r / v.heads) * v.pitch + (r % v.heads) * 0;   /* heads part added by caller */
    }
    int64_t off = 0;
    for (int i = v.nr - 1; i >= 0; --i) { off += (r % v.shp[i]) * v.str[i]; r /= v.shp[i]; }
    return off;
}

__global__ __launch_bounds__(kThreads) void gdn_gated_rmsnorm_kernel(
    const uint8_t* __restrict__ x, uint32_t dtX, R3dRow vx,
    const uint8_t* __restrict__ z, uint32_t dtZ, R3dRow vz, int64_t ch,
    const uint8_t* __restrict__ w, uint32_t dtW,
    uint8_t* __restrict__ o, uint32_t dtO, R3dRow vo,
    float eps, int sigm) {
    __shared__ float s_part[kWarps]; __shared__ float s_sc;
    const int64_t r = blockIdx.x;
    const int64_t ebx = (dtX == RAD_F32) ? 4 : 2, ebz = (dtZ == RAD_F32) ? 4 : 2,
                  ebo = (dtO == RAD_F32) ? 4 : 2;
    const uint8_t* __restrict__ xb = x + r3d_rowbase(vx, r) * ebx;
    uint8_t* __restrict__ ob = o + r3d_rowbase(vo, r) * ebo;
    const int64_t zbe = (vz.nr == -1)
        ? (r / vz.heads) * vz.pitch + (r % vz.heads) * ch
        : r3d_rowbase(vz, r);
    const uint8_t* __restrict__ zb = z + zbe * ebz;

    float ss = 0.0f;
    for (int64_t i = (int64_t)threadIdx.x; i < ch; i += (int64_t)kThreads) {
        const float u = ldt2(xb, dtX, i);
        ss += u * u;
    }
    ss = block_sum(ss, s_part);
    if (threadIdx.x == 0) s_sc = 1.0f / sqrtf(ss / (float)ch + eps);
    __syncthreads();
    const float sc = s_sc;

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
    if (t->rank == 2) {
        if (t->stride[1] != 1) return -3;
        /* ref's rowoff: shape[1] == cols reads the row stride; otherwise the linear
         * fallback r*cols (a view into wider rows is read flat, as ref reads it). */
        *pitch = (t->shape[1] == cols) ? t->stride[0] : cols;
        return 0;
    }
    if (t->rank >= 3) {
        if (t->stride[t->rank - 1] != 1) return -3;
        if (t->shape[t->rank - 1] != cols) { *pitch = cols; return 0; }   /* ref's linear fallback */
        /* shape[last] == cols: the leading-dim walk degenerates to r*cols exactly when
         * the leading dims are contiguous; refuse anything else (ref walks, this kernel
         * class pitches). */
        for (int i = 0; i + 1 < (int)t->rank - 1; ++i)
            if (t->stride[i] != t->shape[i + 1] * t->stride[i + 1]) return -4;
        *pitch = cols;
        return 0;
    }
    return -4;                                  /* rank 0: a degenerate shape, refused */
}

/* Index-tensor reads (i32/u32/i64), same pattern as the attention rows' ldi_w. */
__device__ __forceinline__ int64_t ldi_g(uint32_t dt, const void* p, int64_t i) {
    switch (dt) {
        case RAD_I32: return (int64_t)((const int32_t*)p)[i];
        case RAD_U32: return (int64_t)((const uint32_t*)p)[i];
        case RAD_I64: return ((const int64_t*)p)[i];
        default:      return 0;
    }
}

/* ---- gdn_kkt_solve ----------------------------------------------------------------------
 * A = (I + strict_lower(diag(beta) . K K^T . e^{g_i - g_j}))^{-1}, per chunk, per v head --
 * ref_gdn.cpp's spec, and the 27B chunked path's second op (head_k 128, chunk 64, k/A bf16,
 * beta/g f32, cu i32/i64, M 1..512 in the recorded band).
 *
 * One block per (chunk, sequence, v head): chunks are independent by construction (each one's
 * A reads only its own tokens' k/beta/g), so the chunk index is a free grid axis -- it is what
 * fills the CUs at prefill M, where ref's N*H parallelism would idle most of the card.
 *
 * BIT-EXACT BY CONSTRUCTION against ref's forward substitution, which is why this row skips
 * libr4d's WMMA 16->32->64 blocked merge: ref's inverse chain per column is
 *   A[i][j] = delta(i,j) - sum_{l=j}^{i-1} M[i][l] * A[l][j]   (f32, l ascending, mul-then-sub),
 * and a thread-owns-column-j walk of exactly that chain reproduces every f32 operation in
 * order -- no accumulator rescale, no fragment layout, no reassociation. The M elements
 * themselves are a d-ascending f32 dot over exactly-widened loads plus ref's exact scale
 * chain (beta*dot)*exp(g_i - g_j) on the band's double-rounded exp; the strict-lower triangle
 * is staged to LDS and the substitution then needs no inter-thread cooperation at all --
 * thread j reads only its own column and the shared M rows.
 *
 * LDS: M's strict triangle (2016 f32) + one column per thread (padded to 65 so a warp's
 * per-thread column walks hit distinct banks) ~= 25 KB -> two blocks/CU. ref's chunk loop and
 * per-chunk guards are the grid mapping itself; a block at or past its sequence's end exits
 * empty, and tail chunks (L < chunk) compute and store only the L existing token rows -- full
 * 64-wide A rows, zeros above the diagonal, exactly as ref writes them.
 */
constexpr uint32_t kKktChunk   = 64;   /* the c64 constraint, the libr4d row's and the pack's */
constexpr uint32_t kKktThreads = 64;
constexpr int32_t  kKktTri     = kKktChunk * (kKktChunk - 1) / 2;

__global__ __launch_bounds__(kKktThreads) void gdn_kkt_solve_kernel(
    const uint8_t* __restrict__ k,  uint32_t dtK, int64_t k_s0b, int64_t k_s1b,
    const uint8_t* __restrict__ bt, uint32_t dtB, int64_t b_s0b,
    const uint8_t* __restrict__ g,  uint32_t dtG, int64_t g_s0b,
    uint8_t* __restrict__ A,        uint32_t dtA, int64_t A_s0b, int64_t A_s1b,
    const void* __restrict__ cu, uint32_t dtC,
    int64_t grp, int64_t Kd) {
    const int64_t h = blockIdx.z, n = blockIdx.y;
    const int64_t bos  = ldi_g(dtC, cu, n), slen = ldi_g(dtC, cu, n + 1) - bos;
    const int64_t c0 = (int64_t)blockIdx.x * kKktChunk;
    if (c0 >= slen) return;
    const int64_t L = (slen - c0 < kKktChunk) ? slen - c0 : kKktChunk;
    const int64_t hk = h / grp, tb = bos + c0;
    const uint8_t* __restrict__ kbh = k + hk * k_s1b;

    __shared__ float s_M[kKktTri];                    /* strict lower M[i][j], j < i, at i*(i-1)/2+j */
    __shared__ float s_col[kKktChunk * (kKktChunk + 1)]; /* per-thread column, stride kKktChunk+1 */
    __shared__ float s_g[kKktChunk], s_b[kKktChunk];

    if ((int64_t)threadIdx.x < L) {
        s_g[threadIdx.x] = ldt2(g  + (tb + threadIdx.x) * g_s0b, dtG, h);
        s_b[threadIdx.x] = ldt2(bt + (tb + threadIdx.x) * b_s0b, dtB, h);
    }
    __syncthreads();

    /* Phase 1: the decay-scaled gram's strict triangle. Pairs are flattened row-major over
     * (i, j<i); every pair's dot is its own d-ascending f32 chain, so the flattening order is
     * free -- only the chain WITHIN a pair is arithmetic. */
    const int64_t pairs = L * (L - 1) / 2;
    for (int64_t p = (int64_t)threadIdx.x; p < pairs; p += (int64_t)kKktThreads) {
        int64_t i = (int64_t)((1.0f + sqrtf(1.0f + 8.0f * (float)p)) * 0.5f);
        while (i > 0 && i * (i - 1) / 2 > p) --i;          /* sqrtf is an index decode, not */
        while ((i + 1) * i / 2 <= p) ++i;                  /* arithmetic: make it exact    */
        const int64_t j = p - i * (i - 1) / 2;
        const uint8_t* __restrict__ ri = kbh + (tb + i) * k_s0b;
        const uint8_t* __restrict__ rj = kbh + (tb + j) * k_s0b;
        float acc = 0.0f;
        if (dtK == RAD_F32) {
            const float* __restrict__ fi = (const float*)ri;
            const float* __restrict__ fj = (const float*)rj;
            for (int64_t d = 0; d < Kd; ++d) acc += fi[d] * fj[d];
        } else {
            const uint16_t* __restrict__ hi = (const uint16_t*)ri;
            const uint16_t* __restrict__ hj = (const uint16_t*)rj;
            for (int64_t d = 0; d < Kd; ++d)
                acc += r3d_bf16_to_f32(hi[d]) * r3d_bf16_to_f32(hj[d]);
        }
        s_M[p] = (s_b[i] * acc) * r3d_exp_f(s_g[i] - s_g[j]);   /* ref's (bi*dot)*exp, exact order */
    }
    __syncthreads();

    /* Phase 2: the inverse, thread j owns column j. col[l] is this thread's own earlier value,
     * so the substitution needs no cooperation and no barrier inside the loops. */
    const int64_t jj = (int64_t)threadIdx.x;
    float* __restrict__ col = s_col + jj * (kKktChunk + 1);
    for (int64_t i = jj; i < L; ++i) {
        float v = (i == jj) ? 1.0f : 0.0f;
        const float* __restrict__ mi = s_M + i * (i - 1) / 2;
        for (int64_t l = jj; l < i; ++l) v -= mi[l] * col[l];
        col[i] = v;
    }
    __syncthreads();

    /* Full 64-wide A rows, one row per existing token: zeros above the diagonal (ref writes
     * them too), this column's inverse below it. Threads sweep j, so a row is a coalesced
     * 64-element store. */
    for (int64_t i = 0; i < L; ++i)
        stt2(A + (tb + i) * A_s0b + h * A_s1b, dtA, jj, (i >= jj) ? col[i] : 0.0f);
}

/* ---- gdn_chunk_scan -----------------------------------------------------------------------
 * W = beta.(V - e^g K S^T), V' = A W, O = scale e^g Q S^T + sA V', S = e^{g_last} S + V'^T K' --
 * ref_gdn.cpp's chunked delta-rule scan, the family's fourth member and the 27B's last big
 * device-row hole (head_k=head_v=128, chunk 64, q/k/v/o/A bf16, g/beta f32, h0/ht f32 rank-4
 * state planes, state_idx i32 with a possibly-wider row pitch, scale ~1/sqrt(head_k) per case).
 *
 * Unlike kkt_solve, the chunks are NOT independent: W reads the state the previous chunk left,
 * so the chunk recurrence is serial and there is no free chunk grid axis. One block per
 * (sequence, v head) -- ref's own REF_PAR_FOR(N*H) decomposition -- and the state lives
 * TRANSPOSED, one thread owning one value column e of St[d][e] in registers (head_k=128 f32 =
 * 128 VGPRs/thread at 128 threads, launch_bounds(128,1) leaves the 256-register budget).
 *
 * BIT-EXACT BY CONSTRUCTION, again by walking ref's chains rather than racing them: every
 * ascending-d dot (W's acc, O's inr, the sA row products) is an unrolled f32 chain over the
 * thread's own state registers and exact bf16-widened loads; V' = A W runs the j-ascending
 * chain DESCENDING in i so the in-place W buffer is overwritten only after its last consumer;
 * the state update is one el-multiply before a j-ascending += chain (per element the same
 * order ref writes); ref's bracketings are literal: bc*(v - ge*acc), (scale*exp(dg))*row,
 * vp*(gs*k). The sA row and the staged g/beta are block broadcasts through LDS with barriers
 * outside every predicate -- threads that own no column (e >= head_v) still march through all
 * barriers.
 *
 * LDS: the W->V' buffer (chunk x head_v f32, dynamic, up to 32 KB) + three 64-float broadcasts.
 * A negative state_idx row skips the block exactly as ref skips the (n,h) row -- no o write,
 * no ht write. h0 absent is a zero initial state; ht absent stores nothing; an empty sequence
 * still replays h0 into ht, as ref's loop structure does.
 */
constexpr uint32_t kScanThreads = 128;   /* one value column per thread: head_v <= 128 */
constexpr int32_t  kScanK       = 128;   /* the state column length: head_k == 128, register-bound */
constexpr uint32_t kScanChunk   = 64;    /* the pack's chunk, kkt_solve's and libr4d's */

__global__ __launch_bounds__(kScanThreads, 1) void gdn_chunk_scan_kernel(
    const uint8_t* __restrict__ q,  uint32_t dtQ, int64_t q_s0b, int64_t q_s1b,
    const uint8_t* __restrict__ k,  uint32_t dtK, int64_t k_s0b, int64_t k_s1b,
    const uint8_t* __restrict__ v,  uint32_t dtV, int64_t v_s0b, int64_t v_s1b,
    const uint8_t* __restrict__ A,  uint32_t dtA, int64_t A_s0b, int64_t A_s1b,
    const uint8_t* __restrict__ g,  uint32_t dtG, int64_t g_s0b,
    const uint8_t* __restrict__ bt, uint32_t dtB, int64_t b_s0b,
    const uint8_t* __restrict__ h0, uint32_t dtH, int64_t h_s0b, int64_t h_s1b, int64_t h_s2b,
    uint8_t* __restrict__ o,        uint32_t dtO, int64_t o_s0b, int64_t o_s1b,
    uint8_t* __restrict__ ht,       uint32_t dtT, int64_t t_s0b, int64_t t_s1b, int64_t t_s2b,
    const void* __restrict__ cu, uint32_t dtC,
    const void* __restrict__ sidx, uint32_t dtS, int64_t sw,
    int64_t V, int64_t vW, int64_t grp, float scale) {
    extern __shared__ float s_W[];   /* [kScanChunk, vW]: W, then V' in place (descending i) */
    __shared__ float s_gc[kScanChunk], s_bc[kScanChunk], s_row[kScanChunk];

    const int64_t h = blockIdx.x, n = blockIdx.y, e = threadIdx.x;
    const int64_t bos  = ldi_g(dtC, cu, n), slen = ldi_g(dtC, cu, n + 1) - bos;
    const int64_t sn = sidx ? ldi_g(dtS, sidx, n * sw) : n;
    if (sn < 0) return;                 /* a padded row owns no state; ref skips o and ht too */
    const int64_t hk = h / grp;
    const uint8_t* __restrict__ qb = q + hk * q_s1b;
    const uint8_t* __restrict__ kb = k + hk * k_s1b;
    const uint8_t* __restrict__ vb = v + h  * v_s1b;
    const uint8_t* __restrict__ Ab = A + h  * A_s1b;
    const bool own = e < V;

    float st[kScanK];                   /* the state's column e, d-ascending; owns-guarded */
    if (own) {
        const uint8_t* __restrict__ hb =
            h0 ? h0 + sn * h_s0b + h * h_s1b + e * h_s2b : nullptr;
        #pragma unroll
        for (int d = 0; d < kScanK; ++d) st[d] = hb ? ldt2(hb, dtH, d) : 0.0f;
    }

    for (int64_t c0 = 0; c0 < slen; c0 += kScanChunk) {
        const int64_t L = (slen - c0 < kScanChunk) ? slen - c0 : kScanChunk;
        const int64_t tb = bos + c0;
        if (e < L) {
            s_gc[e] = ldt2(g  + (tb + e) * g_s0b, dtG, h);
            s_bc[e] = ldt2(bt + (tb + e) * b_s0b, dtB, h);
        }
        __syncthreads();

        /* W = bc . (v - e^g K S^T): the d-ascending chain over this thread's column. */
        if (own) {
            for (int64_t j = 0; j < L; ++j) {
                const uint8_t* __restrict__ kr = kb + (tb + j) * k_s0b;
                float acc = 0.0f;
                #pragma unroll
                for (int d = 0; d < kScanK; ++d) acc += ldt2(kr, dtK, d) * st[d];
                const float ge = r3d_exp_f(s_gc[j]);
                s_W[j * vW + e] =
                    s_bc[j] * (ldt2(vb + (tb + j) * v_s0b, dtV, e) - ge * acc);
            }
            /* V' = A W in place, i DESCENDING: row i consumes j <= i, so writing index i
             * only clobbers slots no later (smaller-i) row reads. */
            for (int64_t i = L - 1; i >= 0; --i) {
                const uint8_t* __restrict__ ar = Ab + (tb + i) * A_s0b;
                float acc = 0.0f;
                for (int64_t j = 0; j <= i; ++j) acc += ldt2(ar, dtA, j) * s_W[j * vW + e];
                s_W[i * vW + e] = acc;
            }
        }

        /* O = (scale e^gi) Q S^T + sA V': sA row[j] = (scale e^{gi-gj}) (q_i . k_j) is one
         * broadcast value per j -- computed by thread j, read by every column. Two barriers
         * per row: row ready before reads, reads done before the next row overwrites. */
        for (int64_t i = 0; i < L; ++i) {
            if (e < L) {
                const uint8_t* __restrict__ qr = qb + (tb + i) * q_s0b;
                const uint8_t* __restrict__ kr = kb + (tb + e) * k_s0b;
                float acc = 0.0f;
                #pragma unroll
                for (int d = 0; d < kScanK; ++d)
                    acc += ldt2(qr, dtQ, d) * ldt2(kr, dtK, d);
                s_row[e] = (scale * r3d_exp_f(s_gc[i] - s_gc[e])) * acc;
            }
            __syncthreads();
            if (own) {
                const uint8_t* __restrict__ qr = qb + (tb + i) * q_s0b;
                float inr = 0.0f;
                #pragma unroll
                for (int d = 0; d < kScanK; ++d) inr += ldt2(qr, dtQ, d) * st[d];
                float acc = (scale * r3d_exp_f(s_gc[i])) * inr;
                for (int64_t j = 0; j <= i; ++j) acc += s_row[j] * s_W[j * vW + e];
                stt2(o + (tb + i) * o_s0b + h * o_s1b, dtO, e, acc);
            }
            __syncthreads();
        }

        /* S = e^{g_last} S + V'^T K': the el-multiply first (all d), then one j-ascending
         * += chain per element, vp*(gs*k) with ref's brackets. */
        if (own) {
            const float el = r3d_exp_f(s_gc[L - 1]);
            #pragma unroll
            for (int d = 0; d < kScanK; ++d) st[d] *= el;
            for (int64_t j = 0; j < L; ++j) {
                const float vp = s_W[j * vW + e];
                const float gs = r3d_exp_f(s_gc[L - 1] - s_gc[j]);
                const uint8_t* __restrict__ kr = kb + (tb + j) * k_s0b;
                #pragma unroll
                for (int d = 0; d < kScanK; ++d) st[d] += vp * (gs * ldt2(kr, dtK, d));
            }
        }
    }

    if (ht && own) {
        uint8_t* __restrict__ tb_ht = ht + sn * t_s0b + h * t_s1b + e * t_s2b;
        #pragma unroll
        for (int d = 0; d < kScanK; ++d) stt2(tb_ht, dtT, d, st[d]);
    }
}

/* ---- gdn_conv_update ----------------------------------------------------------------------
 * One block per (sequence, channel): conv_dim blocks per sequence, exactly ref's
 * REF_PAR_FOR(N * conv_dim) decomposition. Shared holds the three tiny broadcasts ref
 * keeps per-channel in a stack frame: the history window, the weight taps, the bias.
 * Block size 64: decode blocks are launch-floored, and a wide block would only idle
 * more threads at slen == 1. */

constexpr uint32_t kConvThreads = 64;
constexpr int32_t  kConvHistMax = 16;   /* ref's REF_CONV_HIST_MAX */

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
     * than cache slen tokens in registers -- the read comes back from L1/L2. width == 1
     * has no state at all: the second loop would index -1, so the whole rewrite is gated
     * on a non-empty window. */
    if (width > 1) {
        for (int64_t i = threadIdx.x; i + 1 < width - 1; i += kConvThreads)
            stt2(csrow, dtCS, i, s_hist[i + 1]);
        for (int64_t t = threadIdx.x; t < slen; t += kConvThreads)
            stt2(csrow, dtCS, t + width - 2, ldt2(x + (bos + t) * x_pb, dtX, c));
    }
}

/* ------------------------------------------------------------------ gdn_recurrent_update
 * The fifth member: decode's counterpart of the chunk scan -- the delta-rule update per token
 * against the paged state cache, the gated rms norm folded into the epilogue. The 27B's LAST
 * device-graph hole (ref_gdn.cpp; 48 of 64 layers run it every decode step).
 *
 * One block per (sequence, v head), ref's own REF_PAR_FOR(N*Hv) decomposition, and the state
 * lives transposed EXACTLY as on chunk_scan: one thread owns one value column e of St[d][e] in
 * 128 f32 registers. The delta rule made that layout nearly barrier-free by accident of math:
 * the update's two d loops read only kn[d] (a block-broadcast vector) into thread e's own
 * column, so the whole state walk -- decay, the S k product, the rank-1 add, the q readout --
 * is register-local per thread. Only two things cross threads: the qk l2 sums (one d chain,
 * thread 0) and the rms sum.
 *
 * BIT-EXACT BY CONSTRUCTION, with one deliberate twist: the rms reduction is NOT block_sum.
 * Ref accumulates sq2 = orow[0]^2 + orow[1]^2 + ... in ascending e, a 128-long SERIAL chain;
 * a tree sum would reassociate it. So the squares stage to LDS and thread 0 walks the chain
 * in ref's order -- 128 cheap LDS reads per token, correctness the library trades no chain for.
 * The qk sums go the same way (thread 0 over the staged q/k rows). Every other chain is one
 * thread's own unrolled d loop in ref's order with ref's literal bracketings: -exp(alog) *
 * softplus(av + dtb), (1/sqrt(sq + l2eps)) * scale, bt*(v - sk2), orow*sc * nw * silu(z) as
 * three separate f32 steps; softplus/sigmoid/exp ride the band's double-rounded r3d_exp_f and
 * r3d_log1p the conv row proved bit-identical.
 *
 * SCOPE: state_form per_candidate only -- the row constrains it, so the selector never asks
 * about an anchor batch (the entry point also refuses one). The out-projection quantiser pair
 * (o_q/o_scale, issued by the 27B decode band) writes ref's q8_group epilogue: one e4m3 group
 * per (token, head) over the o row AS STORED, scale amax/448. A negative sidx entry
 * skips the state write for that token as ref's `so >= 0` guard does; a negative input slot
 * skips the whole (n, h) row -- output included, ref's `continue`. A_log/dt_bias/norm_w are
 * plain-scheme single-plane weights: the row reads them AS STORED (no layout hook; kbench's
 * weight filter declines a hookless device row only above one plane), and ldt2 widens whatever
 * dtype the operand actually carries -- the same dtype-driven read the oracle's ldt performs,
 * which is what keeps bf16-stored weights (this container's spelling) bit-equal on both sides.
 */
constexpr uint32_t kRecThreads = 128;   /* one value column per thread: head_v <= 128 */
constexpr int32_t  kRecK       = 128;   /* head_k register bound: the column IS a register array */

__global__ __launch_bounds__(kRecThreads, 1) void gdn_recurrent_update_kernel(
    const uint8_t* __restrict__ q,  uint32_t dtQ, int64_t q_s0b, int64_t q_s1b,
    const uint8_t* __restrict__ k,  uint32_t dtK, int64_t k_s0b, int64_t k_s1b,
    const uint8_t* __restrict__ v,  uint32_t dtV, int64_t v_s0b, int64_t v_s1b,
    const uint8_t* __restrict__ av, uint32_t dtA, int64_t a_s0b,
    const uint8_t* __restrict__ bv, uint32_t dtB, int64_t b_s0b,
    const uint8_t* __restrict__ alog_t, uint32_t dtAL, int64_t al_s1b,
    const uint8_t* __restrict__ dtb_t,  uint32_t dtDB, int64_t db_s1b,
    uint8_t* __restrict__ st8, uint32_t dtST, int64_t st_s0b, int64_t st_s1b, int64_t st_s2b,
    const void* __restrict__ sidx, uint32_t dtS, int64_t si_s0b, int64_t si_s1b, int64_t sw,
    const void* __restrict__ nacc, uint32_t dtN,
    const void* __restrict__ cu, uint32_t dtC,
    const uint8_t* __restrict__ z, uint32_t dtZ, int64_t z_s0b, int64_t z_s1b,
    const uint8_t* __restrict__ nw, uint32_t dtW, int64_t w_s1b,
    uint8_t* __restrict__ o,        uint32_t dtO, int64_t o_s0b, int64_t o_s1b,
    uint8_t* __restrict__ oq, R3dRow voq, uint8_t* __restrict__ osc, R3dRow vosc,
    int64_t V, int64_t grp, float scale, float l2eps, float neps, float spthr, int sigm) {
    const int64_t h = blockIdx.x, n = blockIdx.y, e = threadIdx.x;
    const int64_t bos  = ldi_g(dtC, cu, n);
    const int64_t slen = ldi_g(dtC, cu, n + 1) - bos;
    if (slen <= 0) return;                            /* per_candidate: ref's continue */
    const int64_t na = nacc ? ldi_g(dtN, nacc, n) : 1;
    const int64_t rd = na > 0 ? na - 1 : 0;
    const int64_t slot_in = ldi_g(dtS, (const uint8_t*)sidx + n * si_s0b +
                            (rd < sw ? rd : sw - 1) * si_s1b, 0);
    if (slot_in < 0) return;                          /* ref: the whole (n,h) row is skipped */

    const int64_t hk = h / grp;
    const uint8_t* __restrict__ qb = q + hk * q_s1b;
    const uint8_t* __restrict__ kb = k + hk * k_s1b;
    const uint8_t* __restrict__ vb = v + h  * v_s1b;
    uint8_t* __restrict__ stb = st8 + h * st_s1b;
    uint8_t* __restrict__ ob  = o + h * o_s1b;
    const uint8_t* __restrict__ zb = z ? z + h * z_s1b : nullptr;
    const bool own = e < V;

    const float alog = alog_t ? ldt2(alog_t + h * al_s1b, dtAL, 0) : 0.0f;
    const float dtb  = dtb_t  ? ldt2(dtb_t  + h * db_s1b, dtDB, 0) : 0.0f;
    /* norm_w rows are V wide while the block runs 128 threads: the idle tail reads column 0
     * instead of walking off the row (its value is discarded). One offset only -- the byte
     * shift OR the ldt2 index, never both, or the row reads element 2e (proved the hard way). */
    const float nwv  = nw ? ldt2(nw + (own ? e : 0) * w_s1b, dtW, 0) : 1.0f;

    __shared__ float s_q[kRecK], s_kn[kRecK];         /* staged q/k, then normalised in place */
    __shared__ float s_orow[kRecThreads];             /* e-ascending squares for the serial rms */
    __shared__ float s_bc[4], s_sc;

    /* The state column, one thread per value channel: a pure copy (vectorised when the row is
     * dense f32 and aligned, which the slot grid at 512-byte strides is for every page layout). */
    float st[kRecK];
    float sv = 0.0f;                 /* the o value AS STORED: ref's quantiser reads it back */
    if (own) {
        const uint8_t* __restrict__ sb = stb + slot_in * st_s0b + e * st_s2b;
        if (dtST == RAD_F32 && ((uintptr_t)sb & 15u) == 0) {
            const float4* __restrict__ s4 = (const float4*)sb;
            #pragma unroll
            for (int d = 0; d < kRecK / 4; ++d) {
                const float4 f = s4[d];
                st[4*d] = f.x; st[4*d+1] = f.y; st[4*d+2] = f.z; st[4*d+3] = f.w;
            }
        } else {
            #pragma unroll
            for (int d = 0; d < kRecK; ++d) st[d] = ldt2(sb, dtST, d);
        }
    }

    for (int64_t t = 0; t < slen; ++t) {
        const int64_t tok = bos + t;
        /* Stage this token's q/k rows cooperatively; the l2 chains then walk the staged values
         * (exact-widened) in ref's d-ascending order from thread 0. gv/bt are thread-invariant
         * scalars computed once beside them. */
        s_q[e]  = ldt2(qb + tok * q_s0b, dtQ, e);
        s_kn[e] = ldt2(kb + tok * k_s0b, dtK, e);
        const float vreg = ldt2(vb + tok * v_s0b, dtV, own ? e : 0);   /* V-wide row clamp */
        __syncthreads();
        if (e == 0) {
            float sq = 0.0f, sk = 0.0f;
            for (int d = 0; d < kRecK; ++d) {
                sq += s_q[d] * s_q[d];
                sk += s_kn[d] * s_kn[d];
            }
            s_bc[0] = 1.0f / sqrtf(sq + l2eps) * scale;      /* ref's (1/sqrt) * scale */
            s_bc[1] = 1.0f / sqrtf(sk + l2eps);
            const float a = ldt2(av + tok * a_s0b, dtA, h);
            s_bc[2] = -r3d_exp_f(alog) * r3d_softplus(a + dtb, spthr);
            s_bc[3] = r3d_sigmoid(ldt2(bv + tok * b_s0b, dtB, h));
        }
        __syncthreads();
        const float qs = s_bc[0], ks = s_bc[1], gv = s_bc[2], bt = s_bc[3];
        s_q[e]  *= qs;                                        /* qn, kn in place */
        s_kn[e] *= ks;
        __syncthreads();

        if (own) {
            /* The delta rule, ref's update(): decay + the S k product in one ascending d loop,
             * the beta correction, then the rank-1 add -- all register-local per column. */
            const float dec = r3d_exp_f(gv);
            float sk2 = 0.0f;
            #pragma unroll
            for (int d = 0; d < kRecK; ++d) { st[d] *= dec; sk2 += st[d] * s_kn[d]; }
            const float uu = bt * (vreg - sk2);
            #pragma unroll
            for (int d = 0; d < kRecK; ++d) st[d] += uu * s_kn[d];

            float orow = 0.0f;                                /* the q readout, ascending d */
            #pragma unroll
            for (int d = 0; d < kRecK; ++d) orow += st[d] * s_q[d];

            s_orow[e] = orow * orow;                          /* squares for the serial rms */
            float sq2 = 0.0f;
            __syncthreads();
            if (e == 0) { sq2 = s_orow[0]; for (int64_t i = 1; i < V; ++i) sq2 += s_orow[i];
                          s_sc = 1.0f / sqrtf(sq2 / (float)V + neps); }
            __syncthreads();
            float r = orow * s_sc;
            r *= nwv;                                         /* ref's three separate steps */
            if (z) {
                const float zz = ldt2(zb + tok * z_s0b, dtZ, e);
                r *= sigm ? r3d_sigmoid(zz) : zz * r3d_sigmoid(zz);
            }
            stt2(ob + tok * o_s0b, dtO, e, r);
            sv = (dtO == RAD_F32) ? r : r3d_bf16_to_f32(r3d_f32_to_bf16(r));
        } else {
            __syncthreads(); __syncthreads();
        }

        if (oq) {
            /* ref's quantiser (ref_common.h q8_group, top 448): one group per (token, head)
             * over the o row AS STORED (bf16 o quantises bf16-rounded values, as ref's ldt
             * after its stt sees). The scale is amax/448, 1.0 for an all-zero group; the codes
             * clamp to +-448 and encode e4m3. Row bases walk ref's rowoff: the head axis is
             * the block's, the channel axis the owning threads'. Block-uniform `oq`, so the
             * barriers here are as legal as the norm's pair above. */
            s_orow[e] = sv < 0.0f ? -sv : sv;      /* idle threads write 0; amax walks < V */
            __syncthreads();
            if (e == 0) {
                float amax = 0.0f;
                for (int64_t i = 0; i < V; ++i) { const float m = s_orow[i]; if (m > amax) amax = m; }
                s_sc = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
            }
            __syncthreads();
            if (own) {
                float c = sv * (1.0f / s_sc);
                if (c > 448.0f) c = 448.0f;
                if (c < -448.0f) c = -448.0f;
                oq[r3d_rowbase(voq, tok) + h * V + e] = r3d_f32_to_e4m3(c);
                if (e == 0) ((float*)(osc + r3d_rowbase(vosc, tok) * 4))[h] = s_sc;
            }
        }

        const int64_t so = ldi_g(dtS, (const uint8_t*)sidx + n * si_s0b +
                             (t < sw ? t : sw - 1) * si_s1b, 0);
        if (own && so >= 0) {
            uint8_t* __restrict__ srow = stb + so * st_s0b + e * st_s2b;
            if (dtST == RAD_F32 && ((uintptr_t)srow & 15u) == 0) {
                float4* __restrict__ s4 = (float4*)srow;
                #pragma unroll
                for (int d = 0; d < kRecK / 4; ++d)
                    s4[d] = make_float4(st[4*d], st[4*d+1], st[4*d+2], st[4*d+3]);
            } else {
                #pragma unroll
                for (int d = 0; d < kRecK; ++d) stt2(srow, dtST, d, st[d]);
            }
        }
        __syncthreads();                                       /* next token rewrites s_q/s_orow */
    }
}

/* ------------------------------------------------------------------ gdn_conv_prep
 * The prefill-side counterpart of conv_update (ref_gdn.cpp): the same depthwise conv with
 * silu writing q/k/v over the sequence window (the conv state read as RAW history, and
 * rewritten from the sequence's last width-1 tokens once the conv has consumed it), the q/k
 * per-head l2 norms, and the gate cumsum with beta. Ref's four phases are three launches
 * here: the state commit folds into the conv kernel (each block stages its channel's history
 * in LDS before any write lands), and the norm and the gate get their own grids.
 *
 * Every chain walks ref's order: the conv is the width-tap f32 chain conv_update proved
 * bit-exact against the same window reads; the l2 sum walks the staged conv output -- which
 * made the round trip through the output dtype, exactly what ref's re-read of the stored
 * bf16 sees -- serially in ascending d at thread 0, tree sums would reassociate it; and the
 * cumsum resets at every chunk boundary, so one (head, sequence, chunk) block walks ref's
 * serial chain over at most chunk tokens while the chunk itself is a free grid axis.
 * exp/softplus/sigmoid are the band's double-rounded r3d_* helpers.
 */
constexpr uint32_t kPrepThreads = 64;         /* conv grid: threads per channel */
constexpr uint32_t kPrepNormThreads = 256;    /* l2 grid: threads over one head row */

__global__ __launch_bounds__(kPrepThreads) void gdn_conv_prep_conv_kernel(
    const uint8_t* __restrict__ x, uint32_t dtX, int64_t x_pb,
    const uint8_t* __restrict__ w, uint32_t dtW, int64_t w_pb,
    const uint8_t* __restrict__ b, uint32_t dtB,
    uint8_t* __restrict__ cs, uint32_t dtCS, int64_t cs_s0b, int64_t cs_s1b,
    const void* __restrict__ sidx, uint32_t dtS,
    const void* __restrict__ hini, uint32_t dtI,
    const void* __restrict__ cu, uint32_t dtC,
    uint8_t* __restrict__ q, uint32_t dtQ, int64_t q_s0b, int64_t q_s1b,
    uint8_t* __restrict__ k, uint32_t dtK, int64_t k_s0b, int64_t k_s1b,
    uint8_t* __restrict__ v, uint32_t dtV, int64_t v_s0b, int64_t v_s1b,
    int64_t HgK, int64_t Kd, int64_t K2, int64_t Vd,
    int32_t width, int64_t cs_slots)
{
    const int64_t c = blockIdx.x, n = blockIdx.y;
    const int64_t bos  = ldi_g(dtC, cu, n);
    const int64_t slen = ldi_g(dtC, cu, n + 1) - bos;
    if (slen <= 0) return;
    const int64_t slot = sidx ? ldi_g(dtS, sidx, n) : n;
    const bool warm = !hini || ldi_g(dtI, hini, n) != 0;
    const bool cs_on = cs && slot >= 0 && slot < cs_slots;
    uint8_t* __restrict__ csrow = cs_on ? cs + slot * cs_s0b + c * cs_s1b : nullptr;

    __shared__ float s_hist[kConvHistMax];
    __shared__ float s_w[kConvHistMax];
    __shared__ float s_b;
    for (int32_t j = (int32_t)threadIdx.x; j < width - 1; j += (int32_t)kPrepThreads)
        s_hist[j] = (warm && csrow) ? ldt2(csrow, dtCS, j) : 0.0f;  /* cold slot: zero window, not recycled bytes */
    for (int32_t j = (int32_t)threadIdx.x; j < width; j += (int32_t)kPrepThreads)
        s_w[j] = ldt2(w + c * w_pb, dtW, j);
    if (threadIdx.x == 0) s_b = b ? ldt2(b, dtB, c) : 0.0f;
    __syncthreads();

    const int32_t w1 = width - 1;
    uint8_t* ob; uint32_t dtO; int64_t s0b, s1b;
    if (c < HgK)       { ob = q; dtO = dtQ; s0b = q_s0b; s1b = q_s1b; }
    else if (c < K2)   { ob = k; dtO = dtK; s0b = k_s0b; s1b = k_s1b; }
    else               { ob = v; dtO = dtV; s0b = v_s0b; s1b = v_s1b; }
    const int64_t h = (c < HgK) ? c / Kd : (c < K2) ? (c - HgK) / Kd : (c - K2) / Vd;
    const int64_t d = (c < K2) ? c % Kd : (c - K2) % Vd;
    uint8_t* __restrict__ orow = ob + h * s1b;

    for (int64_t t = threadIdx.x; t < slen; t += kPrepThreads) {
        float acc = s_b;
        for (int32_t j = 0; j < width; ++j) {
            const int32_t src = (int32_t)t - w1 + j;
            const float xv = src >= 0 ? ldt2(x + (bos + src) * x_pb, dtX, c) : s_hist[src + w1];
            acc += xv * s_w[j];
        }
        stt2(orow + (bos + t) * s0b, dtO, d, acc * r3d_sigmoid(acc));
    }

    /* Ref's phase 4: the state holds the sequence's last width-1 tokens; a shorter sequence
     * keeps the old state's head (zeros when cold). The keeps stage through s_hist (dead
     * after the conv) before ANY write, because a shift's reads and writes overlap the row. */
    if (csrow && width > 1) {
        for (int32_t i = (int32_t)threadIdx.x; i < width - 1; i += (int32_t)kPrepThreads) {
            const int64_t src = slen - w1 + i;
            s_hist[i] = src >= 0 ? ldt2(x + (bos + src) * x_pb, dtX, c)
                                 : (warm ? ldt2(csrow, dtCS, src + w1) : 0.0f);
        }
        __syncthreads();
        for (int32_t i = (int32_t)threadIdx.x; i < width - 1; i += (int32_t)kPrepThreads)
            stt2(csrow, dtCS, i, s_hist[i]);
    }
}

__global__ __launch_bounds__(kPrepNormThreads) void gdn_conv_prep_norm_kernel(
    uint8_t* __restrict__ q, uint32_t dtQ, int64_t q_s0b, int64_t q_s1b,
    uint8_t* __restrict__ k, uint32_t dtK, int64_t k_s0b, int64_t k_s1b,
    int64_t Kd, float l2eps)
{
    extern __shared__ float s_row[];                          /* one head row: Kd floats */
    const int64_t t = blockIdx.x, h = blockIdx.y;
    const bool is_k = blockIdx.z != 0;
    uint8_t* __restrict__ row = is_k ? k + t * k_s0b + h * k_s1b
                                     : q + t * q_s0b + h * q_s1b;
    const uint32_t dt = is_k ? dtK : dtQ;
    for (int64_t i = threadIdx.x; i < Kd; i += blockDim.x) s_row[i] = ldt2(row, dt, i);
    __syncthreads();
    __shared__ float s_inv;
    if (threadIdx.x == 0) {                                   /* ref's serial d chain */
        float ss = 0.0f;
        for (int64_t d = 0; d < Kd; ++d) ss += s_row[d] * s_row[d];
        s_inv = 1.0f / sqrtf(ss + l2eps);
    }
    __syncthreads();
    for (int64_t i = threadIdx.x; i < Kd; i += blockDim.x)
        stt2(row, dt, i, s_row[i] * s_inv);                   /* one store rounding, as ref's stt */
}

__global__ __launch_bounds__(32) void gdn_conv_prep_gate_kernel(
    const uint8_t* __restrict__ x, uint32_t dtX, int64_t x_pb,
    const uint8_t* __restrict__ ta, uint32_t dtA, int64_t a_s0b,
    const uint8_t* __restrict__ tb, uint32_t dtBt, int64_t b_s0b,
    const uint8_t* __restrict__ alog_t, uint32_t dtAL,
    const uint8_t* __restrict__ dtb_t, uint32_t dtDB,
    uint8_t* __restrict__ g, uint32_t dtG, int64_t g_s0b,
    uint8_t* __restrict__ bt, uint32_t dtBt2, int64_t bt_s0b,
    const void* __restrict__ cu, uint32_t dtC,
    int64_t conv_dim, int64_t H, int32_t chunk, float spthr)
{
    const int64_t h = blockIdx.x, n = blockIdx.y;
    const int64_t c0 = (int64_t)blockIdx.z * chunk;
    const int64_t bos  = ldi_g(dtC, cu, n);
    const int64_t slen = ldi_g(dtC, cu, n + 1) - bos;
    if (slen <= 0 || c0 >= slen) return;
    if (threadIdx.x) return;                    /* the cumsum is serial; warp idle, no barriers */
    const float alog = alog_t ? ldt2(alog_t, dtAL, h) : 0.0f;
    const float dtb  = dtb_t  ? ldt2(dtb_t, dtDB, h) : 0.0f;
    const float neg_exp = -r3d_exp_f(alog);     /* ref's -ref_exp(alog), the negation after rounding */
    const int64_t tend = (slen - c0 < chunk) ? slen : c0 + chunk;
    float run = 0.0f;                           /* every chunk starts cold: ref's t % chunk == 0 reset */
    for (int64_t t = c0; t < tend; ++t) {
        const float av_ = ta ? ldt2(ta + (bos + t) * a_s0b, dtA, h)
                             : ldt2(x + (bos + t) * x_pb, dtX, conv_dim + h);
        const float bvl = tb ? ldt2(tb + (bos + t) * b_s0b, dtBt, h)
                             : ldt2(x + (bos + t) * x_pb, dtX, conv_dim + H + h);
        run += neg_exp * r3d_softplus(av_ + dtb, spthr);
        stt2(g  + (bos + t) * g_s0b,  dtG,   h, run);
        stt2(bt + (bos + t) * bt_s0b, dtBt2, h, r3d_sigmoid(bvl));
    }
}

/* ------------------------------------------------------------------ gdn_conv_recurrent_update
 * The decode-band pair fusion (ref_gdn.cpp: conv_update into a staging buffer, then
 * recurrent_update over it) in ONE launch, libr4d's shape: the per-(sequence, v head) block
 * convolves its group's q/k heads and its own v head into LDS -- rounded through x's dtype
 * by stt2/ldt2, so the staging round trip the two-op path pays through global memory happens
 * identically here -- commits the conv window, then walks the shipped recurrent chain with
 * the staged rows in place of the staging tensors. Bit-exact against ref's fused op by the
 * same construction as the two rows it replaces, because it IS the two rows' code paths,
 * fused.
 *
 * The conv commit lands BEFORE the state_idx skip check, in ref's order: the pair writes the
 * conv state even when the recurrent leg then skips the row. Group-sibling blocks re-commit
 * the same q/k channels with byte-identical values -- a benign identical-value race that
 * libr4d solves with an arrival counter; here the redundancy costs one small store loop.
 * CONV_WIDTH up to 4 (the window lives in registers; the 27B band is width 4 -- ref's CONV_
 * HIST_MAX is 16 and the conv_update row still carries it). The conv guard failures that
 * conv_update skips per sequence skip the fused row whole, commit included: ref's composite
 * would fail the OP at that point (it propagates conv_update's error), and the library's
 * documented per-sequence skip stands for the fused form too -- no recorded case has one.
 */
constexpr uint32_t kFusWidthMax = 4;

__global__ __launch_bounds__(kRecThreads, 1) void gdn_conv_recurrent_update_kernel(
    const uint8_t* __restrict__ x, uint32_t dtX, int64_t x_pb,
    const uint8_t* __restrict__ w, uint32_t dtW, int64_t w_pb,
    const uint8_t* __restrict__ cb, uint32_t dtCB,
    uint8_t* __restrict__ cs, uint32_t dtCS, int64_t cs_s0b, int64_t cs_s1b,
    int64_t cs_slots, int64_t state_len,
    const void* __restrict__ cidx, uint32_t dtCI,
    const uint8_t* __restrict__ av, uint32_t dtA, int64_t a_s0b,
    const uint8_t* __restrict__ bv, uint32_t dtB, int64_t b_s0b,
    const uint8_t* __restrict__ alog_t, uint32_t dtAL, int64_t al_s1b,
    const uint8_t* __restrict__ dtb_t,  uint32_t dtDB, int64_t db_s1b,
    uint8_t* __restrict__ st8, uint32_t dtST, int64_t st_s0b, int64_t st_s1b, int64_t st_s2b,
    const void* __restrict__ sidx, uint32_t dtS, int64_t si_s0b, int64_t si_s1b, int64_t sw,
    const void* __restrict__ nacc, uint32_t dtN,
    const void* __restrict__ cu, uint32_t dtC,
    const uint8_t* __restrict__ z, uint32_t dtZ, int64_t z_s0b, int64_t z_s1b,
    const uint8_t* __restrict__ nw, uint32_t dtNW, int64_t w_s1b,
    uint8_t* __restrict__ o,        uint32_t dtO, int64_t o_s0b, int64_t o_s1b,
    uint8_t* __restrict__ oq, R3dRow voq, uint8_t* __restrict__ osc, R3dRow vosc,
    int64_t HgK, int64_t K2, int64_t V, int64_t grp, int32_t width,
    float scale, float l2eps, float neps, float spthr, int sigm) {
    const int64_t h = blockIdx.x, n = blockIdx.y, e = threadIdx.x;
    const int64_t bos  = ldi_g(dtC, cu, n);
    const int64_t slen = ldi_g(dtC, cu, n + 1) - bos;
    if (slen <= 0) return;
    const int64_t na = nacc ? ldi_g(dtN, nacc, n) : 1;
    const int64_t rd = na > 0 ? na - 1 : 0;
    const int64_t cslot = ldi_g(dtCI, cidx, n);
    if (cslot < 0 || cslot >= cs_slots) return;               /* conv_update's skip, fused */
    if (rd < 0 || rd + width - 1 > state_len) return;
    if (width > 1 && width - 2 + slen > state_len) return;

    const int64_t hk = h / grp;
    const int64_t cq = hk * kRecK + e, ck = HgK + cq;         /* e < head_k == 128 always */
    const bool own_v = e < V;
    const int64_t cv = 2 * HgK + h * V + e;
    uint8_t* __restrict__ csq = cs + cslot * cs_s0b + cq * cs_s1b;
    uint8_t* __restrict__ csk = cs + cslot * cs_s0b + ck * cs_s1b;
    uint8_t* __restrict__ csv = own_v ? cs + cslot * cs_s0b + cv * cs_s1b : nullptr;

    const uint32_t eb = (dtX == RAD_F32) ? 4u : 2u;
    extern __shared__ uint8_t s_stg[];   /* slen*(2*head_k + V) elements of x's dtype */
    uint8_t* __restrict__ stq = s_stg;
    uint8_t* __restrict__ stk = stq + slen * kRecK * eb;
    uint8_t* __restrict__ stv = stk + slen * kRecK * eb;

    float hq[kFusWidthMax], hkk[kFusWidthMax], hv[kFusWidthMax];
    float wq[kFusWidthMax], wk[kFusWidthMax], wvv[kFusWidthMax];
    const float bq = cb ? ldt2(cb, dtCB, cq) : 0.0f;
    const float bk = cb ? ldt2(cb, dtCB, ck) : 0.0f;
    const float bvv = cb ? ldt2(cb, dtCB, cv) : 0.0f;
    for (int32_t j = 0; j < width - 1; ++j) {
        hq[j] = ldt2(csq, dtCS, rd + j);
        hkk[j] = ldt2(csk, dtCS, rd + j);
        if (own_v) hv[j] = ldt2(csv, dtCS, rd + j);
    }
    for (int32_t j = 0; j < width; ++j) {
        wq[j] = ldt2(w + cq * w_pb, dtW, j);
        wk[j] = ldt2(w + ck * w_pb, dtW, j);
        if (own_v) wvv[j] = ldt2(w + cv * w_pb, dtW, j);
    }
    __syncthreads();                       /* every history read before any window commit */

    const int32_t w1 = width - 1;
    for (int64_t t = 0; t < slen; ++t) {    /* the conv chain, ascending j, ref's bracketing */
        float acc = bq, acck = bk, accv = bvv;
        for (int32_t j = 0; j < width; ++j) {
            const int32_t src = (int32_t)t - w1 + j;
            if (src >= 0) {
                const uint8_t* __restrict__ xr = x + (bos + src) * x_pb;
                acc  += ldt2(xr, dtX, cq) * wq[j];
                acck += ldt2(xr, dtX, ck) * wk[j];
                if (own_v) accv += ldt2(xr, dtX, cv) * wvv[j];
            } else {
                acc  += hq[src + w1] * wq[j];
                acck += hkk[src + w1] * wk[j];
                if (own_v) accv += hv[src + w1] * wvv[j];
            }
        }
        stt2(stq + t * kRecK * eb, dtX, e, acc * r3d_sigmoid(acc));    /* staging as x's dtype */
        stt2(stk + t * kRecK * eb, dtX, e, acck * r3d_sigmoid(acck));
        if (own_v) stt2(stv + t * V * eb, dtX, e, accv * r3d_sigmoid(accv));
    }

    if (width > 1) {                       /* window commit: shifted tail, then this step */
        for (int32_t i = 0; i < w1 - 1; ++i) {
            stt2(csq, dtCS, i, hq[i + 1]);
            stt2(csk, dtCS, i, hkk[i + 1]);
            if (own_v) stt2(csv, dtCS, i, hv[i + 1]);
        }
        for (int64_t t = 0; t < slen; ++t) {
            const int64_t idx = t + width - 2;
            const uint8_t* __restrict__ xr = x + (bos + t) * x_pb;
            stt2(csq, dtCS, idx, ldt2(xr, dtX, cq));
            stt2(csk, dtCS, idx, ldt2(xr, dtX, ck));
            if (own_v) stt2(csv, dtCS, idx, ldt2(xr, dtX, cv));
        }
    }

    /* ---- the recurrent leg, verbatim the shipped kernel with the staged rows ---- */
    const int64_t slot_in = ldi_g(dtS, (const uint8_t*)sidx + n * si_s0b +
                            (rd < sw ? rd : sw - 1) * si_s1b, 0);
    if (slot_in < 0) return;               /* ref's continue -- AFTER the conv commit above */
    uint8_t* __restrict__ stb = st8 + h * st_s1b;
    uint8_t* __restrict__ ob  = o + h * o_s1b;
    const uint8_t* __restrict__ zb = z ? z + h * z_s1b : nullptr;
    const bool own = e < V;

    __shared__ float s_q[kRecK], s_kn[kRecK];
    __shared__ float s_orow[kRecThreads];
    __shared__ float s_bc[4], s_sc;

    float st[kRecK];
    float sv = 0.0f;                 /* the o value AS STORED: ref's quantiser reads it back */
    if (own) {
        const uint8_t* __restrict__ sb = stb + slot_in * st_s0b + e * st_s2b;
        if (dtST == RAD_F32 && ((uintptr_t)sb & 15u) == 0) {
            const float4* __restrict__ s4 = (const float4*)sb;
            #pragma unroll
            for (int d = 0; d < kRecK / 4; ++d) {
                const float4 f = s4[d];
                st[4*d] = f.x; st[4*d+1] = f.y; st[4*d+2] = f.z; st[4*d+3] = f.w;
            }
        } else {
            #pragma unroll
            for (int d = 0; d < kRecK; ++d) st[d] = ldt2(sb, dtST, d);
        }
    }

    const float nwv = nw ? ldt2(nw + (own ? e : 0) * w_s1b, dtNW, 0) : 1.0f;

    for (int64_t t = 0; t < slen; ++t) {
        const int64_t tok = bos + t;
        s_q[e]  = ldt2(stq + t * kRecK * eb, dtX, e);
        s_kn[e] = ldt2(stk + t * kRecK * eb, dtX, e);
        const float vreg = ldt2(stv + t * V * eb, dtX, own ? e : 0);
        __syncthreads();
        if (e == 0) {
            float sq = 0.0f, sk = 0.0f;
            for (int d = 0; d < kRecK; ++d) {
                sq += s_q[d] * s_q[d];
                sk += s_kn[d] * s_kn[d];
            }
            s_bc[0] = 1.0f / sqrtf(sq + l2eps) * scale;
            s_bc[1] = 1.0f / sqrtf(sk + l2eps);
            const float a = ldt2(av + tok * a_s0b, dtA, h);
            s_bc[2] = -r3d_exp_f(alog_t ? ldt2(alog_t + h * al_s1b, dtAL, 0) : 0.0f)
                      * r3d_softplus(a + (dtb_t ? ldt2(dtb_t + h * db_s1b, dtDB, 0) : 0.0f), spthr);
            s_bc[3] = r3d_sigmoid(ldt2(bv + tok * b_s0b, dtB, h));
        }
        __syncthreads();
        const float qs = s_bc[0], ks = s_bc[1], gv = s_bc[2], bt = s_bc[3];
        s_q[e]  *= qs;
        s_kn[e] *= ks;
        __syncthreads();

        if (own) {
            const float dec = r3d_exp_f(gv);
            float sk2 = 0.0f;
            #pragma unroll
            for (int d = 0; d < kRecK; ++d) { st[d] *= dec; sk2 += st[d] * s_kn[d]; }
            const float uu = bt * (vreg - sk2);
            #pragma unroll
            for (int d = 0; d < kRecK; ++d) st[d] += uu * s_kn[d];

            float orow = 0.0f;
            #pragma unroll
            for (int d = 0; d < kRecK; ++d) orow += st[d] * s_q[d];

            s_orow[e] = orow * orow;
            float sq2 = 0.0f;
            __syncthreads();
            if (e == 0) { sq2 = s_orow[0]; for (int64_t i = 1; i < V; ++i) sq2 += s_orow[i];
                          s_sc = 1.0f / sqrtf(sq2 / (float)V + neps); }
            __syncthreads();
            float r = orow * s_sc;
            r *= nwv;
            if (z) {
                const float zz = ldt2(zb + tok * z_s0b, dtZ, e);
                r *= sigm ? r3d_sigmoid(zz) : zz * r3d_sigmoid(zz);
            }
            stt2(ob + tok * o_s0b, dtO, e, r);
            sv = (dtO == RAD_F32) ? r : r3d_bf16_to_f32(r3d_f32_to_bf16(r));
        } else {
            __syncthreads(); __syncthreads();
        }

        if (oq) {
            /* ref's quantiser (ref_common.h q8_group, top 448): one group per (token, head)
             * over the o row AS STORED (bf16 o quantises bf16-rounded values, as ref's ldt
             * after its stt sees). The scale is amax/448, 1.0 for an all-zero group; the codes
             * clamp to +-448 and encode e4m3. Row bases walk ref's rowoff: the head axis is
             * the block's, the channel axis the owning threads'. Block-uniform `oq`, so the
             * barriers here are as legal as the norm's pair above. */
            s_orow[e] = sv < 0.0f ? -sv : sv;      /* idle threads write 0; amax walks < V */
            __syncthreads();
            if (e == 0) {
                float amax = 0.0f;
                for (int64_t i = 0; i < V; ++i) { const float m = s_orow[i]; if (m > amax) amax = m; }
                s_sc = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
            }
            __syncthreads();
            if (own) {
                float c = sv * (1.0f / s_sc);
                if (c > 448.0f) c = 448.0f;
                if (c < -448.0f) c = -448.0f;
                oq[r3d_rowbase(voq, tok) + h * V + e] = r3d_f32_to_e4m3(c);
                if (e == 0) ((float*)(osc + r3d_rowbase(vosc, tok) * 4))[h] = s_sc;
            }
        }

        const int64_t so = ldi_g(dtS, (const uint8_t*)sidx + n * si_s0b +
                             (t < sw ? t : sw - 1) * si_s1b, 0);
        if (own && so >= 0) {
            uint8_t* __restrict__ srow = stb + so * st_s0b + e * st_s2b;
            if (dtST == RAD_F32 && ((uintptr_t)srow & 15u) == 0) {
                float4* __restrict__ s4 = (float4*)srow;
                #pragma unroll
                for (int d = 0; d < kRecK / 4; ++d)
                    s4[d] = make_float4(st[4*d], st[4*d+1], st[4*d+2], st[4*d+3]);
            } else {
                #pragma unroll
                for (int d = 0; d < kRecK; ++d) stt2(srow, dtST, d, st[d]);
            }
        }
        __syncthreads();
    }
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
    if (rows <= 0 || rows > 0x1000000) return RAD_E_SHAPE;
    if (rad_tensor_numel(o) < rows * ch || rad_tensor_numel(z) < rows * ch) return RAD_E_SHAPE;
    if (rad_tensor_numel(w) < ch) return RAD_E_SHAPE;
    if (w->rank != 1 || w->stride[0] != 1) return RAD_E_STRIDE;

    /* Row addressing per ref's rowoff: last dim == ch walks the leading dims with their own
     * strides (a [T,H,V] read as [T*H,V] stays correct when H's stride is not V); the linear
     * fallback covers rank-1 and last-dim != ch. Within-row unit stride required (as every
     * row here demands). */
    auto rms_view = [&](const RadTensor* t, R3dRow* v) -> int {
        std::memset(v, 0, sizeof *v);
        if (!t->data || t->rank == 0) return -1;
        if (t->stride[t->rank - 1] != 1) return -2;
        if (t->rank >= 2 && t->shape[t->rank - 1] == ch) {
            const int nr = (int)t->rank - 1;
            if (nr > 3) return -3;
            for (int i = 0; i < nr; ++i) {
                if (t->shape[i] <= 0) return -4;
                v->shp[i] = (int32_t)t->shape[i]; v->str[i] = t->stride[i];
            }
            v->nr = nr;
            return 0;
        }
        v->nr = 0; v->pitch = ch;
        return 0;
    };
    R3dRow vx{}, vz{}, vo{};
    if (rms_view(x, &vx)) return RAD_E_STRIDE;
    if (rms_view(o, &vo)) return RAD_E_STRIDE;
    /* The gate: ref's gate_rowoff slice first -- rank-2 whose row is heads*ch wide, read
     * heads-at-a-time -- else the same walk. A rank-1 z is the single-row case. */
    if (z->rank == 2 && z->shape[1] != ch && z->shape[1] % ch == 0 && z->stride[1] == 1) {
        vz.nr = -1; vz.heads = (int32_t)(z->shape[1] / ch); vz.pitch = z->stride[0];
    } else if (rms_view(z, &vz)) return RAD_E_STRIDE;

    gdn_gated_rmsnorm_kernel<<<dim3((uint32_t)rows), dim3(kThreads), 0,
                               (hipStream_t)(uintptr_t)s>>>(
        (const uint8_t*)x->data, x->dtype, vx,
        (const uint8_t*)z->data, z->dtype, vz, ch,
        (const uint8_t*)w->data, w->dtype,
        (uint8_t*)o->data, o->dtype, vo,
        eps, sigm);
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

/* gdn_kkt_solve: verbatim libref operand list (k, beta, g, cu in; A out). The derivation
 * rule (H = numel(beta)/T, Hg = numel(k)/(T*head_k)) is ref's, and so is chunk's fallback
 * to A's trailing shape when the derived parameter is absent. */
extern "C" int r3d_rad_gdn_kkt_solve(const RadArgs* a, RadStream s) {
    const RadTensor* k   = rad_arg_in(a, 0);
    const RadTensor* bt  = rad_arg_in(a, 1);
    const RadTensor* g   = rad_arg_in(a, 2);
    const RadTensor* cu  = rad_arg_in(a, 3);
    const RadTensor* A   = rad_arg_in(a, 4);
    if (!k || !bt || !g || !cu || !A) return RAD_E_INVAL;

    const auto dt2_ok = [](uint32_t d) { return d == RAD_F32 || d == RAD_BF16; };
    if (!dt2_ok(k->dtype) || !dt2_ok(bt->dtype) || !dt2_ok(g->dtype) || !dt2_ok(A->dtype))
        return RAD_E_DTYPE;
    if (!idx_dt_ok(cu->dtype) || !idx_dense(cu)) return RAD_E_STRIDE;

    long long K_ll = 0;
    if (!rad_args_geti(a, "head_k", &K_ll)) return RAD_E_SHAPE;
    const int64_t Kd = K_ll;
    if (Kd <= 0) return RAD_E_SHAPE;

    /* chunk is RAD_DERIVED: present when the issuer supplies it, else ref reads it from
     * A's trailing shape. This kernel is LDS-bound at 64 and refuses the rest. */
    long long ch_ll = 0;
    if (!rad_args_geti(a, "chunk", &ch_ll))
        ch_ll = A->rank >= 3 ? A->shape[A->rank - 1] : 0;
    const int64_t chunk = ch_ll;
    if (chunk <= 0) return RAD_E_SHAPE;
    if (chunk != kKktChunk) return RAD_E_UNSUPPORTED;

    const int64_t N = rad_tensor_numel(cu) - 1;
    if (N <= 0 || N > 0xffff) return RAD_E_SHAPE;
    /* T = the token axis: k->shape[rank-3] at rank >= 3 (v3's token stride), shape[0] on a
     * rank-2 flattening. NOT shape[rank-2], which is the HEAD axis here -- the conv row's
     * reading is rank-2-specific, and kbench's guard allocations caught the mixup as a 256 B
     * write past A on the first L==chunk cases. */
    const int64_t T = k->rank >= 3 ? k->shape[k->rank - 3] : (k->rank >= 2 ? k->shape[0] : 1);
    if (T <= 0) return RAD_E_SHAPE;
    const int64_t H  = rad_tensor_numel(bt) / T;
    const int64_t Hg = rad_tensor_numel(k) / (T * Kd);
    if (H <= 0 || Hg <= 0 || H % Hg != 0) return RAD_E_SHAPE;
    if (rad_tensor_numel(k) < T * Hg * Kd) return RAD_E_SHAPE;
    if (rad_tensor_numel(bt) < T * H || rad_tensor_numel(g) < T * H) return RAD_E_SHAPE;
    if (rad_tensor_numel(A) < T * H * chunk) return RAD_E_SHAPE;
    const int64_t chunks = (T + chunk - 1) / chunk;   /* the x axis: per-seq counts never exceed it */
    if (chunks > 0xffff || H > 0xffff) return RAD_E_SHAPE;

    /* Strides verbatim ref's views (v3(k, Hg, K), v2(beta, H), v2(g, H), v3(A, H, chunk)):
     * tensor strides at the matching rank, tight below; the unit-stride axis must be unit. */
    const int64_t k_s0 = k->rank >= 3 ? k->stride[k->rank - 3] : Hg * Kd;
    const int64_t k_s1 = k->rank >= 3 ? k->stride[k->rank - 2] : Kd;
    if (k->rank >= 3 && k->stride[k->rank - 1] != 1) return RAD_E_STRIDE;
    const int64_t b_s0 = bt->rank >= 2 ? bt->stride[bt->rank - 2] : H;
    const int64_t b_s1 = bt->rank >= 2 ? bt->stride[bt->rank - 1] : 1;
    if (b_s1 != 1) return RAD_E_STRIDE;
    const int64_t g_s0 = g->rank >= 2 ? g->stride[g->rank - 2] : H;
    const int64_t g_s1 = g->rank >= 2 ? g->stride[g->rank - 1] : 1;
    if (g_s1 != 1) return RAD_E_STRIDE;
    const int64_t A_s0 = A->rank >= 3 ? A->stride[A->rank - 3] : H * chunk;
    const int64_t A_s1 = A->rank >= 3 ? A->stride[A->rank - 2] : chunk;
    if (A->rank >= 3 && A->stride[A->rank - 1] != 1) return RAD_E_STRIDE;

    gdn_kkt_solve_kernel<<<dim3((uint32_t)chunks, (uint32_t)N, (uint32_t)H),
                           dim3(kKktThreads), 0, (hipStream_t)(uintptr_t)s>>>(
        (const uint8_t*)k->data,  k->dtype,  rad_dtype_bytes(k->dtype, k_s0),  rad_dtype_bytes(k->dtype, k_s1),
        (const uint8_t*)bt->data, bt->dtype, rad_dtype_bytes(bt->dtype, b_s0),
        (const uint8_t*)g->data,  g->dtype,  rad_dtype_bytes(g->dtype, g_s0),
        (uint8_t*)A->data, A->dtype, rad_dtype_bytes(A->dtype, A_s0), rad_dtype_bytes(A->dtype, A_s1),
        cu->data, cu->dtype,
        H / Hg, Kd);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* gdn_chunk_scan: verbatim libref operand list (q, k, v, A, g, beta, h0?, cu, o, ht?,
 * state_idx?). The derivation rules are ref's (H = numel(beta)/T, Hg = numel(q)/(T*head_k),
 * T from q's own token axis at rank-3 -- the kkt lesson: shape[rank-2] is the HEAD axis on
 * these operands); scale is the caller's f64 parameter read as f32 like ref's p_f32, and the
 * recorded packs all carry 1/sqrt(head_k). h0/ht strides are READ off the tensor at rank >= 4,
 * never derived: ref's v4 comment says a paged state cache is padded to the attention page.
 * state_idx rows may be wider than one slot (speculative decode keeps extras beside column 0);
 * the scan reads column 0 at stride[0] and a negative slot skips the whole (n, h) row --
 * o included, exactly as ref's `continue` does. `state_fp16` is accepted and ignored, as ref
 * ignores it. */
extern "C" int r3d_rad_gdn_chunk_scan(const RadArgs* a, RadStream s) {
    const RadTensor* q    = rad_arg_in(a, 0);
    const RadTensor* k    = rad_arg_in(a, 1);
    const RadTensor* v    = rad_arg_in(a, 2);
    const RadTensor* A    = rad_arg_in(a, 3);
    const RadTensor* g    = rad_arg_in(a, 4);
    const RadTensor* bt   = rad_arg_in(a, 5);
    const RadTensor* h0   = rad_arg_in(a, 6);        /* optional: absent is a zero initial state */
    const RadTensor* cu   = rad_arg_in(a, 7);
    const RadTensor* o    = rad_arg_in(a, 8);
    const RadTensor* ht   = rad_arg_in(a, 9);        /* optional: absent stores no state */
    const RadTensor* sidx = rad_arg_in(a, 10);       /* optional: slots pool is not batch-ordered */
    if (!q || !k || !v || !A || !g || !bt || !cu || !o) return RAD_E_INVAL;

    const auto dt2_ok = [](uint32_t d) { return d == RAD_F32 || d == RAD_BF16; };
    const bool h0_on = h0 && h0->data, ht_on = ht && ht->data;
    const void* sidx_p = (sidx && sidx->data) ? sidx->data : nullptr;
    if (!dt2_ok(q->dtype) || !dt2_ok(k->dtype) || !dt2_ok(v->dtype) || !dt2_ok(A->dtype) ||
        !dt2_ok(g->dtype) || !dt2_ok(bt->dtype) || !dt2_ok(o->dtype) ||
        (h0_on && !dt2_ok(h0->dtype)) || (ht_on && !dt2_ok(ht->dtype))) return RAD_E_DTYPE;
    if (!idx_dt_ok(cu->dtype) || !idx_dense(cu)) return RAD_E_STRIDE;
    if (sidx_p && !idx_dt_ok(sidx->dtype)) return RAD_E_DTYPE;

    long long K_ll = 0, V_ll = 0;
    if (!rad_args_geti(a, "head_k", &K_ll) || !rad_args_geti(a, "head_v", &V_ll))
        return RAD_E_SHAPE;
    const int64_t Kd = K_ll, Vd = V_ll;
    if (Kd <= 0 || Vd <= 0) return RAD_E_SHAPE;
    /* head_k 128 is the register bound (the state column IS a thread's register array);
     * head_v <= 128 is the one-column-per-thread block. The pack records 128/128. */
    if (Kd != kScanK || Vd > (int64_t)kScanThreads) return RAD_E_UNSUPPORTED;

    long long ch_ll = 0;
    if (!rad_args_geti(a, "chunk", &ch_ll))
        ch_ll = A->rank >= 3 ? A->shape[A->rank - 1] : 0;
    if (ch_ll <= 0) return RAD_E_SHAPE;
    if (ch_ll != kScanChunk) return RAD_E_UNSUPPORTED;

    const int64_t N = rad_tensor_numel(cu) - 1;
    if (N <= 0 || N > 0xffff) return RAD_E_SHAPE;
    const int64_t T = q->rank >= 3 ? q->shape[q->rank - 3] : (q->rank >= 2 ? q->shape[0] : 1);
    if (T <= 0) return RAD_E_SHAPE;
    const int64_t H  = rad_tensor_numel(bt) / T;
    const int64_t Hg = rad_tensor_numel(q) / (T * Kd);
    if (H <= 0 || Hg <= 0 || H % Hg != 0) return RAD_E_SHAPE;
    if (rad_tensor_numel(k) < T * Hg * Kd) return RAD_E_SHAPE;
    if (rad_tensor_numel(v) < T * H * Vd || rad_tensor_numel(o) < T * H * Vd) return RAD_E_SHAPE;
    if (rad_tensor_numel(A) < T * H * kScanChunk) return RAD_E_SHAPE;
    if (rad_tensor_numel(g) < T * H || rad_tensor_numel(bt) < T * H) return RAD_E_SHAPE;

    int64_t sw = 1;
    if (sidx_p) {
        if (sidx->rank < 1 || sidx->shape[0] < N) return RAD_E_SHAPE;
        sw = sidx->stride[0];
    } else {
        const int64_t need = N * H * Vd * Kd;
        if (ht_on && rad_tensor_numel(ht) < need) return RAD_E_SHAPE;
        if (h0_on && rad_tensor_numel(h0) < need) return RAD_E_SHAPE;
    }

    /* Strides verbatim ref's views: v3(q/k, Hg, K), v3(v/o, H, V), v3(A, H, chunk), v2(g/beta,
     * H), v4(h0/ht, H, V, K); tensor strides at the matching rank, tight below; the unit-stride
     * axis must be unit (ref's at3/4 accept wider, this library refuses them everywhere). */
    auto v3_ok = [&](const RadTensor* t, int64_t heads, int64_t dim,
                     int64_t* s0, int64_t* s1) -> int {
        *s0 = t->rank >= 3 ? t->stride[t->rank - 3] : heads * dim;
        *s1 = t->rank >= 3 ? t->stride[t->rank - 2] : dim;
        if (t->rank >= 3 && t->stride[t->rank - 1] != 1) return RAD_E_STRIDE;
        return RAD_OK;
    };
    int64_t q_s0, q_s1, k_s0, k_s1, v_s0, v_s1, A_s0, A_s1, o_s0, o_s1;
    int rc = v3_ok(q, Hg, Kd, &q_s0, &q_s1); if (rc) return rc;
    rc = v3_ok(k, Hg, Kd, &k_s0, &k_s1);     if (rc) return rc;
    rc = v3_ok(v, H, Vd, &v_s0, &v_s1);      if (rc) return rc;
    rc = v3_ok(A, H, kScanChunk, &A_s0, &A_s1); if (rc) return rc;
    rc = v3_ok(o, H, Vd, &o_s0, &o_s1);      if (rc) return rc;

    const int64_t b_s0 = bt->rank >= 2 ? bt->stride[bt->rank - 2] : H;
    if (bt->rank >= 2 && bt->stride[bt->rank - 1] != 1) return RAD_E_STRIDE;
    const int64_t g_s0 = g->rank >= 2 ? g->stride[g->rank - 2] : H;
    if (g->rank >= 2 && g->stride[g->rank - 1] != 1) return RAD_E_STRIDE;

    auto v4_strides = [&](const RadTensor* t, int64_t* s0, int64_t* s1, int64_t* s2) -> int {
        *s0 = t->rank >= 4 ? t->stride[t->rank - 4] : H * Vd * Kd;
        *s1 = t->rank >= 4 ? t->stride[t->rank - 3] : Vd * Kd;
        *s2 = t->rank >= 4 ? t->stride[t->rank - 2] : Kd;
        if (t->rank >= 4 && t->stride[t->rank - 1] != 1) return RAD_E_STRIDE;
        return RAD_OK;
    };
    int64_t h_s0 = 0, h_s1 = 0, h_s2 = 0, t_s0 = 0, t_s1 = 0, t_s2 = 0;
    if (h0_on && v4_strides(h0, &h_s0, &h_s1, &h_s2)) return RAD_E_STRIDE;
    if (ht_on && v4_strides(ht, &t_s0, &t_s1, &t_s2)) return RAD_E_STRIDE;

    const double scale_d = rad_args_getf_or(a, "scale", 1.0);
    const float scale = (float)scale_d;

    const uint32_t shm = (uint32_t)(kScanChunk * Vd * sizeof(float));
    const uint32_t dtH = h0_on ? h0->dtype : RAD_F32, dtT = ht_on ? ht->dtype : RAD_F32;
    gdn_chunk_scan_kernel<<<dim3((uint32_t)H, (uint32_t)N), dim3(kScanThreads), shm,
                            (hipStream_t)(uintptr_t)s>>>(
        (const uint8_t*)q->data, q->dtype, rad_dtype_bytes(q->dtype, q_s0), rad_dtype_bytes(q->dtype, q_s1),
        (const uint8_t*)k->data, k->dtype, rad_dtype_bytes(k->dtype, k_s0), rad_dtype_bytes(k->dtype, k_s1),
        (const uint8_t*)v->data, v->dtype, rad_dtype_bytes(v->dtype, v_s0), rad_dtype_bytes(v->dtype, v_s1),
        (const uint8_t*)A->data, A->dtype, rad_dtype_bytes(A->dtype, A_s0), rad_dtype_bytes(A->dtype, A_s1),
        (const uint8_t*)g->data,  g->dtype,  rad_dtype_bytes(g->dtype, g_s0),
        (const uint8_t*)bt->data, bt->dtype, rad_dtype_bytes(bt->dtype, b_s0),
        h0_on ? (const uint8_t*)h0->data : nullptr, dtH,
        rad_dtype_bytes(dtH, h_s0), rad_dtype_bytes(dtH, h_s1), rad_dtype_bytes(dtH, h_s2),
        (uint8_t*)o->data, o->dtype, rad_dtype_bytes(o->dtype, o_s0), rad_dtype_bytes(o->dtype, o_s1),
        ht_on ? (uint8_t*)ht->data : nullptr, dtT,
        rad_dtype_bytes(dtT, t_s0), rad_dtype_bytes(dtT, t_s1), rad_dtype_bytes(dtT, t_s2),
        cu->data, cu->dtype,
        sidx_p, sidx_p ? sidx->dtype : 0, sw,
        Vd, Vd, H / Hg, scale);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* gdn_recurrent_update: verbatim libref operand list (q, k, v, a, b, A_log?, dt_bias?, state
 * INOUT, state_idx, num_accepted, cu, z?, norm_w?, o, o_q?, o_scale?). Derivations are ref's
 * (Hv from the stated n_head_v or numel(v)/(T*head_v), T = numel(v)/(Hv*head_v), Hg =
 * numel(q)/(T*head_k)) -- no shape[rank-2] reading anywhere, per the kkt lesson. The three
 * weight operands are read AS STORED through ldt2, dtype-driven exactly like ref's ldt: the
 * 27B container stores them plain bf16, a single plane each, so kbench's weight filter accepts
 * the hookless row and both sides see the same bytes. o_q/o_scale (the folded out-projection
 * quantiser) are honoured when both are present, as ref's epilogue defines them. A paged state
 * cache keeps its strides read off the tensor at rank >= 4 (page padding included). */
/* The quantiser pair (o_q, o_scale), ref's contract: enabled only when BOTH operands carry
 * data (a lone member is ignored, exactly as ref's `if (oq && osc)` reads it), the fp8 plane
 * is e4m3 and the scale plane f32, both unit last-stride (the library-wide refusal of wider
 * unit axes). Both are read through ref's rowoff view: last extent == cols walks the leading
 * dims with their own strides, everything else is the flat r*cols fallback. */
static int r3d_qpair_view(const RadTensor* oq, const RadTensor* osc, int64_t T, int64_t Hv,
                          int64_t Vd, bool* on, R3dRow* voq, R3dRow* vosc) {
    *on = false;
    std::memset(voq, 0, sizeof *voq); std::memset(vosc, 0, sizeof *vosc);
    if (!oq || !oq->data || !osc || !osc->data) return RAD_OK;
    if (oq->dtype != RAD_F8E4M3 || osc->dtype != RAD_F32) return RAD_E_DTYPE;
    if (oq->rank < 1 || osc->rank < 1) return RAD_E_SHAPE;
    if (oq->stride[oq->rank - 1] != 1 || osc->stride[osc->rank - 1] != 1) return RAD_E_STRIDE;
    if (rad_tensor_numel(oq) < T * Hv * Vd || rad_tensor_numel(osc) < T * Hv) return RAD_E_SHAPE;
    const RadTensor* ts[2] = { oq, osc };
    const int64_t cols[2] = { Hv * Vd, Hv };
    R3dRow* vs[2] = { voq, vosc };
    for (int s = 0; s < 2; ++s) {
        const RadTensor* t = ts[s];
        if (t->rank >= 2 && t->shape[t->rank - 1] == cols[s]) {     /* ref's rowoff walk */
            if (t->rank > 4) return RAD_E_UNSUPPORTED;
            R3dRow* v = vs[s];
            v->nr = (int32_t)(t->rank - 1);
            int64_t rows = 1;
            for (int i = 0; i < v->nr; ++i) {
                v->shp[i] = (int32_t)t->shape[i]; v->str[i] = t->stride[i];
                rows *= t->shape[i];
            }
            if (rows < T) return RAD_E_SHAPE;
        } else {
            vs[s]->nr = 0; vs[s]->pitch = cols[s];                  /* ref's linear fallback */
        }
    }
    *on = true;
    return RAD_OK;
}

extern "C" int r3d_rad_gdn_recurrent_update(const RadArgs* a, RadStream s) {
    const RadTensor* q    = rad_arg_in(a, 0);
    const RadTensor* k    = rad_arg_in(a, 1);
    const RadTensor* v    = rad_arg_in(a, 2);
    const RadTensor* av   = rad_arg_in(a, 3);
    const RadTensor* bv   = rad_arg_in(a, 4);
    const RadTensor* alog = rad_arg_in(a, 5);        /* optional; ref defaults the element to 0 */
    const RadTensor* dtb  = rad_arg_in(a, 6);        /* optional */
    const RadTensor* st   = rad_arg_in(a, 7);
    const RadTensor* sidx = rad_arg_in(a, 8);
    const RadTensor* nacc = rad_arg_in(a, 9);
    const RadTensor* cu   = rad_arg_in(a, 10);
    const RadTensor* z    = rad_arg_in(a, 11);       /* optional: absent means no gate */
    const RadTensor* nw   = rad_arg_in(a, 12);       /* optional */
    const RadTensor* o    = rad_arg_in(a, 13);
    const RadTensor* oq   = rad_arg_in(a, 14);       /* optional fp8 quantiser pair */
    const RadTensor* osc  = rad_arg_in(a, 15);
    /* num_accepted is schema-required but a drafter-less serve issues it null; ref reads it
     * null-tolerantly (absent == 1), so this row must not demand it. */
    if (!q || !k || !v || !av || !bv || !st || !sidx || !cu || !o) return RAD_E_INVAL;

    const char* sf = rad_args_gets(a, "state_form");
    if (sf && std::strcmp(sf, "per_candidate") != 0) return RAD_E_UNSUPPORTED;   /* anchor: r4d's */

    const auto dt2_ok = [](uint32_t d) { return d == RAD_F32 || d == RAD_BF16; };
    const bool alog_on = alog && alog->data, dtb_on = dtb && dtb->data;
    const bool z_on = z && z->data, nw_on = nw && nw->data;
    if (!dt2_ok(q->dtype) || !dt2_ok(k->dtype) || !dt2_ok(v->dtype) || !dt2_ok(av->dtype) ||
        !dt2_ok(bv->dtype) || !dt2_ok(st->dtype) || !dt2_ok(o->dtype) ||
        (alog_on && !dt2_ok(alog->dtype)) || (dtb_on && !dt2_ok(dtb->dtype)) ||
        (z_on && !dt2_ok(z->dtype)) || (nw_on && !dt2_ok(nw->dtype))) return RAD_E_DTYPE;
    if (!idx_dt_ok(cu->dtype) || !idx_dense(cu)) return RAD_E_STRIDE;
    if (nacc && (!idx_dt_ok(nacc->dtype) || !idx_dense(nacc))) return RAD_E_STRIDE;
    if (!idx_dt_ok(sidx->dtype)) return RAD_E_DTYPE;

    long long K_ll = 0, V_ll = 0;
    if (!rad_args_geti(a, "head_k", &K_ll) || !rad_args_geti(a, "head_v", &V_ll))
        return RAD_E_SHAPE;
    const int64_t Kd = K_ll, Vd = V_ll;
    if (Kd <= 0 || Vd <= 0) return RAD_E_SHAPE;
    /* head_k 128 is the register bound (the state column IS a thread's register array, as on
     * chunk_scan); head_v <= 128 is the one-column-per-thread block. The pack records 128/128. */
    if (Kd != kRecK || Vd > (int64_t)kRecThreads) return RAD_E_UNSUPPORTED;

    long long hv_ll = 0;
    if (!rad_args_geti(a, "n_head_v", &hv_ll)) hv_ll = 0;
    /* T is ref's derivation numel(v)/(Hv*head_v), Hv stated or from v's head axis -- but a
     * launcher reads shapes, not device data: derive T from the dense shape the same way. */
    const int64_t Hv = hv_ll > 0 ? hv_ll
                                 : (v->rank >= 3 ? v->shape[v->rank - 2] : 0);
    if (Hv <= 0) return RAD_E_SHAPE;
    const int64_t T = rad_tensor_numel(v) / (Hv * Vd);
    if (T <= 0) return RAD_E_SHAPE;
    const int64_t Hg = rad_tensor_numel(q) / (T * Kd);
    if (Hg <= 0 || Hv % Hg != 0) return RAD_E_SHAPE;

    const int64_t N = rad_tensor_numel(cu) - 1;
    if (N <= 0 || N > 0xffff) return RAD_E_SHAPE;
    if (rad_tensor_numel(k) < T * Hg * Kd) return RAD_E_SHAPE;
    if (rad_tensor_numel(o) < T * Hv * Vd) return RAD_E_SHAPE;
    if (rad_tensor_numel(av) < T * Hv || rad_tensor_numel(bv) < T * Hv) return RAD_E_SHAPE;
    if (nacc && rad_tensor_numel(nacc) < N) return RAD_E_SHAPE;

    /* state_idx: ref reads at2(si, n, col) with si = v2(sidx, sw), sw = shape[rank-1]; accept
     * rank >= 2 with a unit last stride and rows >= N. */
    if (sidx->rank < 2 || sidx->stride[sidx->rank - 1] != 1) return RAD_E_STRIDE;
    const int64_t sw    = sidx->shape[sidx->rank - 1];
    const int64_t si_s0 = sidx->stride[sidx->rank - 2];
    const int64_t si_s1 = sidx->stride[sidx->rank - 1];
    if (sw <= 0 || sidx->shape[sidx->rank - 2] < N) return RAD_E_SHAPE;

    const double scale_d = rad_args_getf_or(a, "scale", 1.0);
    const double eps_d   = rad_args_getf_or(a, "eps", 1e-6);
    const double l2_d    = rad_args_getf_or(a, "l2_eps", 1e-6);
    const double thr_d   = rad_args_getf_or(a, "softplus_thr", 20.0);
    const char* act = rad_args_gets(a, "act");
    const int sigm = act && !std::strcmp(act, "sigmoid");
    if (act && !sigm && std::strcmp(act, "silu")) return RAD_E_UNSUPPORTED;

    /* Strides verbatim ref's views: v3(q/k, Hg, K), v3(v/o/z, Hv, V), v2(a/b, Hv),
     * v4(state, Hv, V, K); tensor strides at the matching rank, tight below; unit last
     * stride required everywhere (the library-wide refusal of wider unit axes). */
    auto v3_ok = [&](const RadTensor* t, int64_t heads, int64_t dim,
                     int64_t* s0, int64_t* s1) -> int {
        *s0 = t->rank >= 3 ? t->stride[t->rank - 3] : heads * dim;
        *s1 = t->rank >= 3 ? t->stride[t->rank - 2] : dim;
        if (t->rank >= 3 && t->stride[t->rank - 1] != 1) return RAD_E_STRIDE;
        return RAD_OK;
    };
    int64_t q_s0, q_s1, k_s0, k_s1, v_s0, v_s1, o_s0, o_s1, z_s0 = 0, z_s1 = 0;
    int rc = v3_ok(q, Hg, Kd, &q_s0, &q_s1); if (rc) return rc;
    rc = v3_ok(k, Hg, Kd, &k_s0, &k_s1);     if (rc) return rc;
    rc = v3_ok(v, Hv, Vd, &v_s0, &v_s1);     if (rc) return rc;
    rc = v3_ok(o, Hv, Vd, &o_s0, &o_s1);     if (rc) return rc;
    if (z_on) {
        rc = v3_ok(z, Hv, Vd, &z_s0, &z_s1); if (rc) return rc;
        if (rad_tensor_numel(z) < T * Hv * Vd) return RAD_E_SHAPE;
    }
    const int64_t a_s0 = av->rank >= 2 ? av->stride[av->rank - 2] : Hv;
    if (av->rank >= 2 && av->stride[av->rank - 1] != 1) return RAD_E_STRIDE;
    const int64_t b_s0 = bv->rank >= 2 ? bv->stride[bv->rank - 2] : Hv;
    if (bv->rank >= 2 && bv->stride[bv->rank - 1] != 1) return RAD_E_STRIDE;

    /* state: slots x heads x V x K, strides read off the rank-4 tensor (a paged cache is
     * page-padded on the slot stride; only the innermost unit stride is assumed, as on
     * chunk_scan's h0/ht). */
    const int64_t st_s0 = st->rank >= 4 ? st->stride[st->rank - 4] : Hv * Vd * Kd;
    const int64_t st_s1 = st->rank >= 4 ? st->stride[st->rank - 3] : Vd * Kd;
    const int64_t st_s2 = st->rank >= 4 ? st->stride[st->rank - 2] : Kd;
    if (st->rank >= 4 && st->stride[st->rank - 1] != 1) return RAD_E_STRIDE;
    if (rad_tensor_numel(st) < T * Hv * Vd * Kd) return RAD_E_SHAPE;

    /* 1-D weight vectors: ref's laststride reads; unit last stride (dense vectors). */
    auto vec_ok = [&](const RadTensor* t, int64_t n, int64_t* s1) -> int {
        if (!t) return RAD_OK;
        if (t->rank < 1 || t->stride[t->rank - 1] != 1) return RAD_E_STRIDE;
        if (rad_tensor_numel(t) < n) return RAD_E_SHAPE;
        *s1 = 1;
        return RAD_OK;
    };
    int64_t al_s1 = 1, db_s1 = 1, w_s1 = 1;
    rc = vec_ok(alog_on ? alog : nullptr, Hv, &al_s1); if (rc) return rc;
    rc = vec_ok(dtb_on ? dtb : nullptr, Hv, &db_s1);   if (rc) return rc;
    rc = vec_ok(nw_on ? nw : nullptr, Vd, &w_s1);      if (rc) return rc;

    bool q_on = false; R3dRow voq, vosc;
    int rq = r3d_qpair_view(oq, osc, T, Hv, Vd, &q_on, &voq, &vosc); if (rq) return rq;

    gdn_recurrent_update_kernel<<<dim3((uint32_t)Hv, (uint32_t)N), dim3(kRecThreads), 0,
                                  (hipStream_t)(uintptr_t)s>>>(
        (const uint8_t*)q->data, q->dtype, rad_dtype_bytes(q->dtype, q_s0), rad_dtype_bytes(q->dtype, q_s1),
        (const uint8_t*)k->data, k->dtype, rad_dtype_bytes(k->dtype, k_s0), rad_dtype_bytes(k->dtype, k_s1),
        (const uint8_t*)v->data, v->dtype, rad_dtype_bytes(v->dtype, v_s0), rad_dtype_bytes(v->dtype, v_s1),
        (const uint8_t*)av->data, av->dtype, rad_dtype_bytes(av->dtype, a_s0),
        (const uint8_t*)bv->data, bv->dtype, rad_dtype_bytes(bv->dtype, b_s0),
        alog_on ? (const uint8_t*)alog->data : nullptr, alog_on ? alog->dtype : 0,
        alog_on ? rad_dtype_bytes(alog->dtype, al_s1) : 0,
        dtb_on ? (const uint8_t*)dtb->data : nullptr, dtb_on ? dtb->dtype : 0,
        dtb_on ? rad_dtype_bytes(dtb->dtype, db_s1) : 0,
        (uint8_t*)st->data, st->dtype,
        rad_dtype_bytes(st->dtype, st_s0), rad_dtype_bytes(st->dtype, st_s1),
        rad_dtype_bytes(st->dtype, st_s2),
        sidx->data, sidx->dtype, rad_dtype_bytes(sidx->dtype, si_s0),
        rad_dtype_bytes(sidx->dtype, si_s1), sw,
        nacc ? nacc->data : nullptr, nacc ? nacc->dtype : 0,
        cu->data, cu->dtype,
        z_on ? (const uint8_t*)z->data : nullptr, z_on ? z->dtype : 0,
        z_on ? rad_dtype_bytes(z->dtype, z_s0) : 0, z_on ? rad_dtype_bytes(z->dtype, z_s1) : 0,
        nw_on ? (const uint8_t*)nw->data : nullptr, nw_on ? nw->dtype : 0,
        nw_on ? rad_dtype_bytes(nw->dtype, w_s1) : 0,
        (uint8_t*)o->data, o->dtype, rad_dtype_bytes(o->dtype, o_s0), rad_dtype_bytes(o->dtype, o_s1),
        q_on ? (uint8_t*)oq->data : nullptr, voq, q_on ? (uint8_t*)osc->data : nullptr, vosc,
        Vd, Hv / Hg, (float)scale_d, (float)l2_d, (float)eps_d, (float)thr_d, sigm);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* gdn_conv_prep: verbatim libref operand list (x, w, b?, A_log?, dt_bias?, conv_state INOUT,
 * cu, a?, b_gate?, cache_idx?, has_init?, q, k, v, g, beta). Derivations are ref's (Hg =
 * numel(q)/(T*head_k), H = numel(v)/(T*head_v), conv_dim = 2*Hg*K + H*V). T is the row count x
 * carries: ref reads cu's last entry, device data a launcher cannot see -- the conv_update
 * precedent, same number for every valid call. chunk drives the gate cumsum reset and the
 * chunked grid; the row refuses anything but the pack's 64. Absent a/b_gate, ref slices the
 * last 2*H columns of the fused x row. Every weight is read AS STORED through dtype-driven
 * ldt2 reads -- plain single-plane in this container, no layout hook, same discovery that
 * unblocked the recurrent row. The three launches (conv+commit, l2 norms, gate) preserve ref's
 * phase order: norm reads what the conv stored, exactly as ref's re-read does. */
extern "C" int r3d_rad_gdn_conv_prep(const RadArgs* a, RadStream s) {
    const RadTensor* x    = rad_arg_in(a, 0);
    const RadTensor* w    = rad_arg_in(a, 1);
    const RadTensor* b    = rad_arg_in(a, 2);        /* optional */
    const RadTensor* alog = rad_arg_in(a, 3);        /* optional data */
    const RadTensor* dtb  = rad_arg_in(a, 4);        /* optional data */
    const RadTensor* cs   = rad_arg_in(a, 5);        /* INOUT; null data = cold, no commit */
    const RadTensor* cu   = rad_arg_in(a, 6);
    const RadTensor* ta   = rad_arg_in(a, 7);        /* optional, all-or-nothing with tb */
    const RadTensor* tb   = rad_arg_in(a, 8);
    const RadTensor* sidx = rad_arg_in(a, 9);        /* optional: cache_idx */
    const RadTensor* hini = rad_arg_in(a, 10);       /* optional: has_init */
    const RadTensor* q    = rad_arg_in(a, 11);
    const RadTensor* k    = rad_arg_in(a, 12);
    const RadTensor* v    = rad_arg_in(a, 13);
    const RadTensor* g    = rad_arg_in(a, 14);
    const RadTensor* bt   = rad_arg_in(a, 15);
    if (!x || !w || !cu || !q || !k || !v || !g || !bt) return RAD_E_INVAL;
    if ((ta && !tb) || (!ta && tb)) return RAD_E_INVAL;         /* ref's own refusal */

    const auto dt2_ok = [](uint32_t d) { return d == RAD_F32 || d == RAD_BF16; };
    const bool b_on = b && b->data, alog_on = alog && alog->data, dtb_on = dtb && dtb->data;
    const bool cs_on = cs && cs->data, sidx_on = sidx && sidx->data, hini_on = hini && hini->data;
    if (!dt2_ok(x->dtype) || !dt2_ok(w->dtype) || !dt2_ok(q->dtype) || !dt2_ok(k->dtype) ||
        !dt2_ok(v->dtype) || !dt2_ok(g->dtype) || !dt2_ok(bt->dtype) ||
        (b_on && !dt2_ok(b->dtype)) || (alog_on && !dt2_ok(alog->dtype)) ||
        (dtb_on && !dt2_ok(dtb->dtype)) || (cs_on && !dt2_ok(cs->dtype)) ||
        (ta && ta->data && !dt2_ok(ta->dtype)) || (tb && tb->data && !dt2_ok(tb->dtype)))
        return RAD_E_DTYPE;
    if (!idx_dt_ok(cu->dtype) || !idx_dense(cu)) return RAD_E_STRIDE;
    if (sidx_on && (!idx_dt_ok(sidx->dtype) || !idx_dense(sidx))) return RAD_E_STRIDE;
    if (hini_on && (!idx_dt_ok(hini->dtype) || !idx_dense(hini))) return RAD_E_STRIDE;

    long long K_ll = 0, V_ll = 0, W_ll = 0, ch_ll = 0;
    if (!rad_args_geti(a, "head_k", &K_ll) || !rad_args_geti(a, "head_v", &V_ll) ||
        !rad_args_geti(a, "conv_width", &W_ll) || !rad_args_geti(a, "chunk", &ch_ll))
        return RAD_E_SHAPE;
    const int64_t Kd = K_ll, Vd = V_ll, width = W_ll, chunk = ch_ll;
    if (Kd <= 0 || Vd <= 0 || width <= 0 || chunk <= 0) return RAD_E_SHAPE;
    if (width - 1 > kConvHistMax || chunk != 64) return RAD_E_UNSUPPORTED;
    if (Kd > 4096) return RAD_E_UNSUPPORTED;          /* the l2 row stages one head into LDS */

    const float l2eps = (float)rad_args_getf_or(a, "l2_eps", 1e-6);
    const float spthr = (float)rad_args_getf_or(a, "softplus_thr", 20.0);

    const int64_t N = rad_tensor_numel(cu) - 1;
    if (N <= 0 || N > 0xffff) return RAD_E_SHAPE;
    const int64_t T = x->rank >= 2 ? x->shape[x->rank - 2] : 1;   /* rank-2 x: the token axis */
    if (T <= 0 || T > 0x100000) return RAD_E_SHAPE;
    const int64_t Hg = rad_tensor_numel(q) / (T * Kd);
    const int64_t H  = rad_tensor_numel(v) / (T * Vd);
    if (Hg <= 0 || H <= 0 || H % Hg != 0) return RAD_E_SHAPE;
    if (Hg > 0xffff || H > 0xffff) return RAD_E_SHAPE;
    const int64_t conv_dim = 2 * Hg * Kd + H * Vd;
    if (rad_tensor_numel(w) < conv_dim * width) return RAD_E_SHAPE;

    /* x: [T, conv_dim], or ref's fused [T, conv_dim + 2H] when the gates ride along. */
    const bool ta_on = ta && ta->data, tb_on = tb && tb->data;
    if (ta_on != tb_on) return RAD_E_INVAL;
    const int64_t xneed = conv_dim + (ta_on ? 0 : 2 * H);
    const int64_t xrow = x->rank >= 2 ? x->stride[x->rank - 2] : xneed;
    if (x->rank >= 2) {
        if (x->stride[x->rank - 1] != 1) return RAD_E_STRIDE;
        /* Bound by the VIEW, as ref does: `x` may be the left conv_dim columns of a wider
         * projection row (sliced at issue, like the fp8 block's fused [q|k|v|z] projection),
         * whose numel undercounts. Only a flattened operand is bounded by numel. */
        if (x->shape[x->rank - 1] < xneed || x->shape[x->rank - 2] < T) return RAD_E_SHAPE;
    } else {
        if (x->stride[0] != 1) return RAD_E_STRIDE;
        if (rad_tensor_numel(x) < (T - 1) * xrow + xneed) return RAD_E_SHAPE;
    }

    /* w: v2(w, width); b: element c of a dense row. */
    const int64_t w_s0 = w->rank >= 2 ? w->stride[w->rank - 2] : width;
    if (w->rank >= 2 && w->stride[w->rank - 1] != 1) return RAD_E_STRIDE;
    if (b_on && (b->rank == 0 || b->stride[b->rank - 1] != 1)) return RAD_E_STRIDE;
    if (b_on && rad_tensor_numel(b) < conv_dim) return RAD_E_SHAPE;

    /* gate sources: separate v2(a/b, H) rows, or the fused x columns; g/beta are v2 rows. */
    const int64_t a_s0 = ta_on ? (ta->rank >= 2 ? ta->stride[ta->rank - 2] : H) : H;
    if (ta_on && ta->rank >= 2 && ta->stride[ta->rank - 1] != 1) return RAD_E_STRIDE;
    const int64_t b_s0 = tb_on ? (tb->rank >= 2 ? tb->stride[tb->rank - 2] : H) : H;
    if (tb_on && tb->rank >= 2 && tb->stride[tb->rank - 1] != 1) return RAD_E_STRIDE;
    if (ta_on && rad_tensor_numel(ta) < T * H) return RAD_E_SHAPE;
    if (tb_on && rad_tensor_numel(tb) < T * H) return RAD_E_SHAPE;
    const int64_t g_s0 = g->rank >= 2 ? g->stride[g->rank - 2] : H;
    if (g->rank >= 2 && g->stride[g->rank - 1] != 1) return RAD_E_STRIDE;
    const int64_t bt_s0 = bt->rank >= 2 ? bt->stride[bt->rank - 2] : H;
    if (bt->rank >= 2 && bt->stride[bt->rank - 1] != 1) return RAD_E_STRIDE;
    if (rad_tensor_numel(g) < T * H || rad_tensor_numel(bt) < T * H) return RAD_E_SHAPE;
    if (alog_on && (alog->rank == 0 || alog->stride[alog->rank - 1] != 1)) return RAD_E_STRIDE;
    if (alog_on && rad_tensor_numel(alog) < H) return RAD_E_SHAPE;
    if (dtb_on && (dtb->rank == 0 || dtb->stride[dtb->rank - 1] != 1)) return RAD_E_STRIDE;
    if (dtb_on && rad_tensor_numel(dtb) < H) return RAD_E_SHAPE;

    /* conv state: ref's conv_state_view -- [slots, conv_dim, state_len] with the tensor's
     * strides, or tight-flat below; the RAW history layout, so no window-depth contract. */
    const int64_t state_len = cs_on ? (cs->rank >= 3 ? cs->shape[cs->rank - 1] : width - 1) : 0;
    if (cs_on && state_len < width - 1) return RAD_E_SHAPE;
    const int64_t cs_s0 = cs_on ? (cs->rank >= 3 ? cs->stride[cs->rank - 3] : conv_dim * state_len) : 0;
    const int64_t cs_s1 = cs_on ? (cs->rank >= 3 ? cs->stride[cs->rank - 2] : state_len) : 0;
    const int64_t cs_s2 = cs_on ? (cs->rank >= 3 ? cs->stride[cs->rank - 1] : 1) : 1;
    if (cs_on && cs_s2 != 1) return RAD_E_STRIDE;
    const int64_t cs_slots = cs_on
        ? (cs->rank >= 3 ? cs->shape[cs->rank - 3] : rad_tensor_numel(cs) / (conv_dim * state_len))
        : 0;
    if (cs_on && cs_slots <= 0) return RAD_E_SHAPE;

    /* q/k/v: ref's v3 views, tensor strides at rank >= 3, tight below, unit last stride. */
    auto out_ok = [&](const RadTensor* t, int64_t heads, int64_t dim,
                      int64_t* s0, int64_t* s1) -> int {
        *s0 = t->rank >= 3 ? t->stride[t->rank - 3] : heads * dim;
        *s1 = t->rank >= 3 ? t->stride[t->rank - 2] : dim;
        if (t->rank >= 3 && t->stride[t->rank - 1] != 1) return RAD_E_STRIDE;
        return rad_tensor_numel(t) < T * heads * dim ? RAD_E_SHAPE : RAD_OK;
    };
    int64_t q_s0, q_s1, k_s0, k_s1, v_s0, v_s1;
    int rc = out_ok(q, Hg, Kd, &q_s0, &q_s1); if (rc) return rc;
    rc = out_ok(k, Hg, Kd, &k_s0, &k_s1);     if (rc) return rc;
    rc = out_ok(v, H, Vd, &v_s0, &v_s1);      if (rc) return rc;

    const int32_t w32 = (int32_t)width, ch32 = (int32_t)chunk;
    const hipStream_t hs = (hipStream_t)(uintptr_t)s;
    gdn_conv_prep_conv_kernel<<<dim3((uint32_t)conv_dim, (uint32_t)N), dim3(kPrepThreads), 0, hs>>>(
        (const uint8_t*)x->data, x->dtype, rad_dtype_bytes(x->dtype, xrow),
        (const uint8_t*)w->data, w->dtype, rad_dtype_bytes(w->dtype, w_s0),
        b_on ? (const uint8_t*)b->data : nullptr, b_on ? b->dtype : 0,
        cs_on ? (uint8_t*)cs->data : nullptr, cs_on ? cs->dtype : 0,
        cs_on ? rad_dtype_bytes(cs->dtype, cs_s0) : 0,
        cs_on ? rad_dtype_bytes(cs->dtype, cs_s1) : 0,
        sidx_on ? sidx->data : nullptr, sidx_on ? sidx->dtype : 0,
        hini_on ? hini->data : nullptr, hini_on ? hini->dtype : 0,
        cu->data, cu->dtype,
        (uint8_t*)q->data, q->dtype, rad_dtype_bytes(q->dtype, q_s0), rad_dtype_bytes(q->dtype, q_s1),
        (uint8_t*)k->data, k->dtype, rad_dtype_bytes(k->dtype, k_s0), rad_dtype_bytes(k->dtype, k_s1),
        (uint8_t*)v->data, v->dtype, rad_dtype_bytes(v->dtype, v_s0), rad_dtype_bytes(v->dtype, v_s1),
        Hg * Kd, Kd, 2 * Hg * Kd, Vd, w32, cs_slots);
    if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;

    const uint32_t shm = (uint32_t)(Kd * sizeof(float));
    gdn_conv_prep_norm_kernel<<<dim3((uint32_t)T, (uint32_t)Hg, 2), dim3(kPrepNormThreads), shm, hs>>>(
        (uint8_t*)q->data, q->dtype, rad_dtype_bytes(q->dtype, q_s0), rad_dtype_bytes(q->dtype, q_s1),
        (uint8_t*)k->data, k->dtype, rad_dtype_bytes(k->dtype, k_s0), rad_dtype_bytes(k->dtype, k_s1),
        Kd, l2eps);
    if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;

    const int64_t chunks = (T + chunk - 1) / chunk;
    if (chunks > 0xffff) return RAD_E_SHAPE;
    gdn_conv_prep_gate_kernel<<<dim3((uint32_t)H, (uint32_t)N, (uint32_t)chunks), dim3(32), 0, hs>>>(
        (const uint8_t*)x->data, x->dtype, rad_dtype_bytes(x->dtype, xrow),
        ta_on ? (const uint8_t*)ta->data : nullptr, ta_on ? ta->dtype : 0,
        ta_on ? rad_dtype_bytes(ta->dtype, a_s0) : 0,
        tb_on ? (const uint8_t*)tb->data : nullptr, tb_on ? tb->dtype : 0,
        tb_on ? rad_dtype_bytes(tb->dtype, b_s0) : 0,
        alog_on ? (const uint8_t*)alog->data : nullptr, alog_on ? alog->dtype : 0,
        dtb_on ? (const uint8_t*)dtb->data : nullptr, dtb_on ? dtb->dtype : 0,
        (uint8_t*)g->data, g->dtype, rad_dtype_bytes(g->dtype, g_s0),
        (uint8_t*)bt->data, bt->dtype, rad_dtype_bytes(bt->dtype, bt_s0),
        cu->data, cu->dtype, conv_dim, H, ch32, spthr);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* gdn_conv_recurrent_update: verbatim libref operand list (x, conv_w, conv_b?, conv_state,
 * conv_idx, a, b, A_log?, dt_bias?, state INOUT, state_idx, num_accepted, cu, z?, norm_w?, o,
 * o_q?, o_scale?) -- the pair's union, staging tensors absent by design. Derivations are ref's:
 * T = o->shape[0], Hv = o->shape[1], conv_dim = x's column count, Hg = (conv_dim - Hv*head_v) /
 * (2*head_k) -- no shape[rank-2] guessing anywhere. The row inherits the recurrent row's scope:
 * head_k 128, head_v <= 128, per_candidate only, quantiser pair refused. conv_width up to 4
 * (the fused window lives in registers). The staging smem is sized from the ONE bound a
 * launcher can see: the conv contract caps a sequence at state_len - conv_width + 2 tokens. */
extern "C" int r3d_rad_gdn_conv_recurrent_update(const RadArgs* a, RadStream s) {
    const RadTensor* x    = rad_arg_in(a, 0);
    const RadTensor* w    = rad_arg_in(a, 1);
    const RadTensor* cb   = rad_arg_in(a, 2);        /* optional */
    const RadTensor* cs   = rad_arg_in(a, 3);
    const RadTensor* cidx = rad_arg_in(a, 4);
    const RadTensor* av   = rad_arg_in(a, 5);
    const RadTensor* bv   = rad_arg_in(a, 6);
    const RadTensor* alog = rad_arg_in(a, 7);        /* optional data */
    const RadTensor* dtb  = rad_arg_in(a, 8);        /* optional data */
    const RadTensor* st   = rad_arg_in(a, 9);
    const RadTensor* sidx = rad_arg_in(a, 10);
    const RadTensor* nacc = rad_arg_in(a, 11);
    const RadTensor* cu   = rad_arg_in(a, 12);
    const RadTensor* z    = rad_arg_in(a, 13);       /* optional */
    const RadTensor* nw   = rad_arg_in(a, 14);       /* optional */
    const RadTensor* o    = rad_arg_in(a, 15);
    const RadTensor* oq   = rad_arg_in(a, 16);       /* optional fp8 quantiser pair */
    const RadTensor* osc  = rad_arg_in(a, 17);
    /* num_accepted: schema-required, issued null by a drafter-less serve. Ref treats the
     * absence as 1; the kernel here does the same (`na = nacc ? load : 1`). */
    if (!x || !w || !cs || !cidx || !av || !bv || !st || !sidx || !cu || !o)
        return RAD_E_INVAL;
    const char* sf = rad_args_gets(a, "state_form");
    if (sf && std::strcmp(sf, "per_candidate") != 0) return RAD_E_UNSUPPORTED;

    const auto dt2_ok = [](uint32_t d) { return d == RAD_F32 || d == RAD_BF16; };
    const bool cb_on = cb && cb->data, alog_on = alog && alog->data, dtb_on = dtb && dtb->data;
    const bool z_on = z && z->data, nw_on = nw && nw->data;
    if (!dt2_ok(x->dtype) || !dt2_ok(w->dtype) || !dt2_ok(cs->dtype) || !dt2_ok(av->dtype) ||
        !dt2_ok(bv->dtype) || !dt2_ok(st->dtype) || !dt2_ok(o->dtype) ||
        (cb_on && !dt2_ok(cb->dtype)) || (alog_on && !dt2_ok(alog->dtype)) ||
        (dtb_on && !dt2_ok(dtb->dtype)) || (z_on && !dt2_ok(z->dtype)) ||
        (nw_on && !dt2_ok(nw->dtype))) return RAD_E_DTYPE;
    if (!idx_dt_ok(cu->dtype) || !idx_dense(cu)) return RAD_E_STRIDE;
    if (nacc && (!idx_dt_ok(nacc->dtype) || !idx_dense(nacc))) return RAD_E_STRIDE;
    if (!idx_dt_ok(cidx->dtype) || !idx_dense(cidx)) return RAD_E_STRIDE;
    if (!idx_dt_ok(sidx->dtype)) return RAD_E_DTYPE;

    long long K_ll = 0, V_ll = 0, W_ll = 0;
    if (!rad_args_geti(a, "head_k", &K_ll) || !rad_args_geti(a, "head_v", &V_ll) ||
        !rad_args_geti(a, "conv_width", &W_ll)) return RAD_E_SHAPE;
    const int64_t Kd = K_ll, Vd = V_ll, width = W_ll;
    if (Kd <= 0 || Vd <= 0 || width <= 0) return RAD_E_SHAPE;
    if (Kd != kRecK || Vd > (int64_t)kRecThreads) return RAD_E_UNSUPPORTED;
    if (width > kFusWidthMax) return RAD_E_UNSUPPORTED;

    if (x->rank < 2 || o->rank != 3 || o->shape[2] != Vd) return RAD_E_SHAPE;   /* ref's checks */
    const int64_t T = o->shape[0], Hv = o->shape[1];
    const int64_t conv_dim = x->shape[x->rank - 1];
    if (T <= 0 || Hv <= 0 || conv_dim <= Hv * Vd) return RAD_E_SHAPE;
    if ((conv_dim - Hv * Vd) % (2 * Kd) != 0) return RAD_E_SHAPE;
    const int64_t Hg = (conv_dim - Hv * Vd) / (2 * Kd);
    if (Hg <= 0 || Hv % Hg != 0) return RAD_E_SHAPE;

    const int64_t N = rad_tensor_numel(cu) - 1;
    if (N <= 0 || N > 0xffff) return RAD_E_SHAPE;
    if (rad_tensor_numel(o) < T * Hv * Vd) return RAD_E_SHAPE;
    if (rad_tensor_numel(av) < T * Hv || rad_tensor_numel(bv) < T * Hv) return RAD_E_SHAPE;
    if ((nacc && rad_tensor_numel(nacc) < N) || rad_tensor_numel(cidx) < N) return RAD_E_SHAPE;
    if (rad_tensor_numel(w) < conv_dim * width) return RAD_E_SHAPE;

    /* x: rows of stride xrow, the left conv_dim columns (a view of a wider projection row is
     * legal -- bound by the view's shape, not its numel, exactly as ref does). */
    const int64_t xrow = x->stride[x->rank - 2];
    if (x->stride[x->rank - 1] != 1) return RAD_E_STRIDE;
    if (x->shape[x->rank - 2] < T) return RAD_E_SHAPE;
    const int64_t w_s0 = w->rank >= 2 ? w->stride[w->rank - 2] : width;
    if (w->rank >= 2 && w->stride[w->rank - 1] != 1) return RAD_E_STRIDE;
    if (cb_on && (cb->rank == 0 || cb->stride[cb->rank - 1] != 1)) return RAD_E_STRIDE;
    if (cb_on && rad_tensor_numel(cb) < conv_dim) return RAD_E_SHAPE;

    /* conv state: ref's conv_state_view (the rolling window, same contract as conv_update). */
    const int64_t state_len = cs->rank >= 3 ? cs->shape[cs->rank - 1] : (width > 1 ? width - 1 : 1);
    const int64_t cs_s0 = cs->rank >= 3 ? cs->stride[cs->rank - 3] : conv_dim * state_len;
    const int64_t cs_s1 = cs->rank >= 3 ? cs->stride[cs->rank - 2] : state_len;
    const int64_t cs_s2 = cs->rank >= 3 ? cs->stride[cs->rank - 1] : 1;
    if (cs_s2 != 1) return RAD_E_STRIDE;
    const int64_t cs_slots = cs->rank >= 3
        ? cs->shape[cs->rank - 3]
        : rad_tensor_numel(cs) / (conv_dim * state_len);
    if (cs_slots <= 0) return RAD_E_SHAPE;

    const int64_t a_s0 = av->rank >= 2 ? av->stride[av->rank - 2] : Hv;
    if (av->rank >= 2 && av->stride[av->rank - 1] != 1) return RAD_E_STRIDE;
    const int64_t b_s0 = bv->rank >= 2 ? bv->stride[bv->rank - 2] : Hv;
    if (bv->rank >= 2 && bv->stride[bv->rank - 1] != 1) return RAD_E_STRIDE;

    /* state / state_idx / z / norm_w / o: the recurrent row's checks, verbatim. */
    if (sidx->rank < 2 || sidx->stride[sidx->rank - 1] != 1) return RAD_E_STRIDE;
    const int64_t sw    = sidx->shape[sidx->rank - 1];
    const int64_t si_s0 = sidx->stride[sidx->rank - 2];
    const int64_t si_s1 = sidx->stride[sidx->rank - 1];
    if (sw <= 0 || sidx->shape[sidx->rank - 2] < N) return RAD_E_SHAPE;

    const int64_t st_s0 = st->rank >= 4 ? st->stride[st->rank - 4] : Hv * Vd * Kd;
    const int64_t st_s1 = st->rank >= 4 ? st->stride[st->rank - 3] : Vd * Kd;
    const int64_t st_s2 = st->rank >= 4 ? st->stride[st->rank - 2] : Kd;
    if (st->rank >= 4 && st->stride[st->rank - 1] != 1) return RAD_E_STRIDE;
    if (rad_tensor_numel(st) < T * Hv * Vd * Kd) return RAD_E_SHAPE;

    auto v3_ok = [&](const RadTensor* t, int64_t heads, int64_t dim,
                     int64_t* s0, int64_t* s1) -> int {
        *s0 = t->rank >= 3 ? t->stride[t->rank - 3] : heads * dim;
        *s1 = t->rank >= 3 ? t->stride[t->rank - 2] : dim;
        if (t->rank >= 3 && t->stride[t->rank - 1] != 1) return RAD_E_STRIDE;
        return RAD_OK;
    };
    int64_t o_s0, o_s1, z_s0 = 0, z_s1 = 0;
    int rc = v3_ok(o, Hv, Vd, &o_s0, &o_s1);  if (rc) return rc;
    if (z_on) {
        rc = v3_ok(z, Hv, Vd, &z_s0, &z_s1);  if (rc) return rc;
        if (rad_tensor_numel(z) < T * Hv * Vd) return RAD_E_SHAPE;
    }
    auto vec_ok = [&](const RadTensor* t, int64_t n, int64_t* s1) -> int {
        if (!t) return RAD_OK;
        if (t->rank < 1 || t->stride[t->rank - 1] != 1) return RAD_E_STRIDE;
        if (rad_tensor_numel(t) < n) return RAD_E_SHAPE;
        *s1 = 1;
        return RAD_OK;
    };
    int64_t al_s1 = 1, db_s1 = 1, w_s1 = 1;
    rc = vec_ok(alog_on ? alog : nullptr, Hv, &al_s1); if (rc) return rc;
    rc = vec_ok(dtb_on ? dtb : nullptr, Hv, &db_s1);   if (rc) return rc;
    rc = vec_ok(nw_on ? nw : nullptr, Vd, &w_s1);      if (rc) return rc;

    bool q_on = false; R3dRow voq, vosc;
    int rq = r3d_qpair_view(oq, osc, T, Hv, Vd, &q_on, &voq, &vosc); if (rq) return rq;

    const double scale_d = rad_args_getf_or(a, "scale", 1.0);
    const double eps_d   = rad_args_getf_or(a, "eps", 1e-6);
    const double l2_d    = rad_args_getf_or(a, "l2_eps", 1e-6);
    const double thr_d   = rad_args_getf_or(a, "softplus_thr", 20.0);
    const char* act = rad_args_gets(a, "act");
    const int sigm = act && !std::strcmp(act, "sigmoid");
    if (act && !sigm && std::strcmp(act, "silu")) return RAD_E_UNSUPPORTED;

    /* The staging bound the conv contract itself draws: width - 2 + slen <= state_len. */
    const int64_t slen_max = (state_len - width + 2) < T ? (state_len - width + 2) : T;
    if (slen_max < 0) return RAD_E_SHAPE;      /* no window can ever fit: caller bug */
    const int64_t eb = (x->dtype == RAD_F32) ? 4 : 2;
    const int64_t smem = slen_max * (2 * Kd + Vd) * eb;
    if (smem > 48 * 1024) return RAD_E_UNSUPPORTED;   /* LDS budget, incl. the recurrent leg */

    const int32_t w32 = (int32_t)width;
    gdn_conv_recurrent_update_kernel<<<dim3((uint32_t)Hv, (uint32_t)N), dim3(kRecThreads),
                                       (uint32_t)smem, (hipStream_t)(uintptr_t)s>>>(
        (const uint8_t*)x->data, x->dtype, rad_dtype_bytes(x->dtype, xrow),
        (const uint8_t*)w->data, w->dtype, rad_dtype_bytes(w->dtype, w_s0),
        cb_on ? (const uint8_t*)cb->data : nullptr, cb_on ? cb->dtype : 0,
        (uint8_t*)cs->data, cs->dtype, rad_dtype_bytes(cs->dtype, cs_s0),
        rad_dtype_bytes(cs->dtype, cs_s1), cs_slots, state_len,
        cidx->data, cidx->dtype,
        (const uint8_t*)av->data, av->dtype, rad_dtype_bytes(av->dtype, a_s0),
        (const uint8_t*)bv->data, bv->dtype, rad_dtype_bytes(bv->dtype, b_s0),
        alog_on ? (const uint8_t*)alog->data : nullptr, alog_on ? alog->dtype : 0,
        alog_on ? rad_dtype_bytes(alog->dtype, al_s1) : 0,
        dtb_on ? (const uint8_t*)dtb->data : nullptr, dtb_on ? dtb->dtype : 0,
        dtb_on ? rad_dtype_bytes(dtb->dtype, db_s1) : 0,
        (uint8_t*)st->data, st->dtype,
        rad_dtype_bytes(st->dtype, st_s0), rad_dtype_bytes(st->dtype, st_s1),
        rad_dtype_bytes(st->dtype, st_s2),
        sidx->data, sidx->dtype, rad_dtype_bytes(sidx->dtype, si_s0),
        rad_dtype_bytes(sidx->dtype, si_s1), sw,
        nacc ? nacc->data : nullptr, nacc ? nacc->dtype : 0,
        cu->data, cu->dtype,
        z_on ? (const uint8_t*)z->data : nullptr, z_on ? z->dtype : 0,
        z_on ? rad_dtype_bytes(z->dtype, z_s0) : 0, z_on ? rad_dtype_bytes(z->dtype, z_s1) : 0,
        nw_on ? (const uint8_t*)nw->data : nullptr, nw_on ? nw->dtype : 0,
        nw_on ? rad_dtype_bytes(nw->dtype, w_s1) : 0,
        (uint8_t*)o->data, o->dtype, rad_dtype_bytes(o->dtype, o_s0), rad_dtype_bytes(o->dtype, o_s1),
        q_on ? (uint8_t*)oq->data : nullptr, voq, q_on ? (uint8_t*)osc->data : nullptr, vosc,
        Hg * Kd, 2 * Hg * Kd, Vd, Hv / Hg, w32,
        (float)scale_d, (float)l2_d, (float)eps_d, (float)thr_d, sigm);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
