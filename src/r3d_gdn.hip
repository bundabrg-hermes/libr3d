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
     * than cache slen tokens in registers -- the read comes back from L1/L2. */
    for (int64_t i = threadIdx.x; i + 1 < width - 1; i += kConvThreads)
        stt2(csrow, dtCS, i, s_hist[i + 1]);
    for (int64_t t = threadIdx.x; t < slen; t += kConvThreads)
        stt2(csrow, dtCS, t + width - 2, ldt2(x + (bos + t) * x_pb, dtX, c));
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
 * (o_q/o_scale, absent from every recorded 27B case) is refused, not ignored: silently
 * skipping it would hand the caller unquantised rows behind its back. A negative sidx entry
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
        } else {
            __syncthreads(); __syncthreads();
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
 * quantiser) are refused, never ignored. A paged state cache keeps its strides read off the
 * tensor at rank >= 4 (page padding included). */
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
    if (!q || !k || !v || !av || !bv || !st || !sidx || !nacc || !cu || !o) return RAD_E_INVAL;
    /* The quantiser pair, all-or-nothing on ref's reading; ours is not implemented, and quietly
     * dropping it would leave the caller's fp8 output plane stale. */
    if ((rad_arg_in(a, 14) && rad_arg_in(a, 14)->data) ||
        (rad_arg_in(a, 15) && rad_arg_in(a, 15)->data)) return RAD_E_UNSUPPORTED;

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
    if (!idx_dt_ok(nacc->dtype) || !idx_dense(nacc)) return RAD_E_STRIDE;
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
    if (rad_tensor_numel(nacc) < N) return RAD_E_SHAPE;

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
        nacc->data, nacc->dtype,
        cu->data, cu->dtype,
        z_on ? (const uint8_t*)z->data : nullptr, z_on ? z->dtype : 0,
        z_on ? rad_dtype_bytes(z->dtype, z_s0) : 0, z_on ? rad_dtype_bytes(z->dtype, z_s1) : 0,
        nw_on ? (const uint8_t*)nw->data : nullptr, nw_on ? nw->dtype : 0,
        nw_on ? rad_dtype_bytes(nw->dtype, w_s1) : 0,
        (uint8_t*)o->data, o->dtype, rad_dtype_bytes(o->dtype, o_s0), rad_dtype_bytes(o->dtype, o_s1),
        Vd, Hv / Hg, (float)scale_d, (float)l2_d, (float)eps_d, (float)thr_d, sigm);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
