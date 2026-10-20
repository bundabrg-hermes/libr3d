/* ============================================================================
 * r3d_gemm_logits.hip -- logits_gemm over a joined e4m3 + block-scale head.
 *
 * The lm_head of the 27B trunk arrives as two selected planes (fp8_e4m3 codes
 * [n_vocab, n_embd] and a bf16 scale per 128x128 block) joined into one blob by
 * the layout hook, exactly the form rad_planes.h defines and ref_gemm's
 * ScaledRows reads: y[m, n] = sum_k f32(x[m,k]) * (f32(code(n,k)) * scale).
 * One rounding in the product of code and scale, one f32 accumulator per
 * output over ascending k -- the reference's own order.
 *
 * No WMMA here on purpose. The decoded weight is not bf16-representable (a
 * 3-bit code times a bf16 scale needs 11+ mantissa bits), so narrowing B to a
 * tensor-core fragment would move every product by ~2^-9 -- ~2e-3 on the sum,
 * right where the tolerance lives. This is a bandwidth row instead: at decode
 * M (1..8) the dot products are noise against streaming 1.27 GB of codes once,
 * so the kernel is a GEMV that reads each code exactly once: warp per head row,
 * lanes striding 4 bytes, x staged per 128-column block through LDS (it is hot
 * either way, but a lane-strided 2-byte walk would double the L2 transactions),
 * and a butterfly fold at the end. The fold reassociates the k chain -- the GEMM
 * house line (r3d_gemm_bf16.hip): rel_l2, not bit-exactness, kbench adjudicates.
 *
 * M derives from numel(y)/N -- never the schema's M parameter (the
 * arena-capacity rule; see r3d_gemm_bf16.hip's account of the 0.35 tok/s run).
 * M beyond the register group rides grid.y: each y-group re-streams the codes,
 * which is why MT tops out at 8 -- a prefill-band M of 64 costs eight passes
 * (~11 ms), still far under the host's scalar lm_head.
 * ========================================================================== */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#include "rad_abi.h"
#include "rad_plugin.h"   /* rad_arg_in / rad_tensor_numel (host-side entry helpers) */
#include "rad_planes.h"

#define R3D_DEVICE __device__ __forceinline__

R3D_DEVICE float bf2f(uint16_t b) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)b << 16;
    return v.f;
}

/* rad_plugin.h rad_fp8e4m3_to_f32, ported verbatim (same rationale as
 * r3d_gemm_mxfp4.hip's copy: one place for the spec to live). */
R3D_DEVICE float fp8e4m3_m4(uint8_t b) {
    const uint32_t s = (uint32_t)(b >> 7) << 31;
    const uint32_t e = (b >> 3) & 0xFu, m = b & 7u;
    union { uint32_t u; float f; } v;
    if (e == 0) {
        if (m == 0) { v.u = s; return v.f; }
        float f = (float)m / 512.0f;             /* m * 2^-9, exact */
        v.u = s | (__float_as_uint(f) & 0x7fffffffu);
        return v.f;
    }
    if (e == 0xF && m == 7) { v.u = s | 0x7fc00000u; return v.f; }   /* NaN */
    v.u = s | ((e - 7u + 127u) << 23) | (m << 20);
    return v.f;
}

/* One scale element of the plane's dtype, widened exactly. */
R3D_DEVICE float ld_scale(const void* base, int64_t idx, uint32_t dt) {
    if (dt == RAD_F32)  return ((const float*)base)[idx];
    if (dt == RAD_F16)  return __half2float(((const __half*)base)[idx]);
    return bf2f(((const uint16_t*)base)[idx]);
}

template <int MT>
__global__ __launch_bounds__(256)
void logits_e4m3_kernel(const uint16_t* __restrict__ X, int64_t x_pitch,
                        const uint8_t* __restrict__ codes, int64_t code_pitch,
                        const uint8_t* __restrict__ scale, int scols, int br, uint32_t sdt,
                        float* __restrict__ Y, int64_t y_pitch,
                        int N, int K, int M, int m_base)
{
    const int w    = threadIdx.x >> 5;         /* warp = one head row */
    const int lane = threadIdx.x & 31;
    const int r    = blockIdx.x * 8 + w;
    if (r >= N) return;

    const int mb   = m_base + blockIdx.y * MT; /* first live x/logits row here */
    const int live = M - mb < MT ? M - mb : MT;

    const int64_t rblk  = (int64_t)r / br;
    const uint8_t* crow = codes + (int64_t)r * code_pitch;

    __shared__ uint16_t xs[MT * 128];          /* one 128-column block of x */
    __shared__ float    s_uni;                 /* the block's scale, staged once */

    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

    const int nfull = K >> 7;
    const int tail  = K & 127;

    for (int jb = 0, jn = nfull + (tail ? 1 : 0); jb < jn; ++jb) {
        const int kbase = jb << 7;
        /* Stage the x block (zero past K) and the scale; two syncs per block,
         * and the staging itself is a perfect 128 B/thread-row walk. */
        const int elems = (jb == jn - 1 && tail) ? tail : 128;
        for (int i = threadIdx.x; i < live * elems; i += 256) {
            const int m = i / elems, k = i - m * elems;
            xs[m * 128 + k] = X[(int64_t)(mb + m) * x_pitch + kbase + k];
        }
        if (threadIdx.x == 0)
            s_uni = ld_scale(scale, rblk * (int64_t)scols + jb, sdt);
        __syncthreads();

        const float s = s_uni;
        const int kl = (jb == jn - 1 && tail) ? tail : 128;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int k = (lane << 2) + t;
            if (k >= kl) break;
            const float cv = fp8e4m3_m4(crow[kbase + k]) * s;
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m < live) acc[m] = __fmaf_rn(bf2f(xs[m * 128 + k]), cv, acc[m]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int off = 16; off; off >>= 1)
            acc[m] += __shfl_xor(acc[m], off);
    }
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < MT; ++m)
            if (m < live) Y[(int64_t)(mb + m) * y_pitch + r] = acc[m];
    }
}

/* ------------------------------------------------------------------ row entry */

extern "C" int r3d_rad_logits_gemm_e4m3(const RadArgs* a, RadStream s) {
    const RadTensor* X = rad_arg_in(a, 0);
    const RadTensor* W = rad_arg_in(a, 1);
    const RadTensor* Y = rad_arg_in(a, 2);
    if (!X || !W || !Y) return RAD_E_UNSUPPORTED;

    const char* dt = rad_args_gets(a, "dtype");
    if (!dt || strcmp(dt, "bf16") != 0) return RAD_E_UNSUPPORTED;
    if (X->dtype != RAD_BF16 || Y->dtype != RAD_F32) return RAD_E_UNSUPPORTED;

    long long N = 0, K = 0;
    if (!rad_args_geti(a, "n_vocab", &N) || !rad_args_geti(a, "n_embd", &K)) return RAD_E_SHAPE;
    if (N <= 0 || K <= 0) return RAD_E_SHAPE;

    /* M is numel(Y)/N -- never the schema's M (the arena-capacity rule). */
    const long long Yn = rad_tensor_numel(Y);
    if (Yn <= 0 || Yn % N) return RAD_E_SHAPE;
    const long long M = Yn / N;
    if (M > 0x7ffffffLL || N > 0x7ffffffLL || K > 0x7ffffffLL) return RAD_E_SHAPE;
    if (rad_tensor_numel(X) < M * K) return RAD_E_SHAPE;
    if ((K & 3) != 0) return RAD_E_UNSUPPORTED;   /* the four-byte lane walk */

    if (X->rank >= 2 && X->stride[X->rank - 1] != 1) return RAD_E_STRIDE;
    if (Y->rank >= 2 && Y->stride[Y->rank - 1] != 1) return RAD_E_STRIDE;
    const int64_t x_pitch = X->rank >= 2 ? X->stride[X->rank - 2] : K;
    const int64_t y_pitch = Y->rank >= 2 ? Y->stride[Y->rank - 2] : N;
    if (x_pitch < K || y_pitch < N) return RAD_E_STRIDE;

    /* The joined planes, checked the way ref's scaled_rows checks them. */
    const RadPlanesHeader* h = rad_planes_header(W->data);
    if (!h || h->n_planes != 2) return RAD_E_DTYPE;
    if (h->transform[0]) return RAD_E_UNSUPPORTED;
    const RadPlanesEntry* ec = nullptr;
    const RadPlanesEntry* es = nullptr;
    const void* pc = rad_planes_find(W->data, "codes", &ec);
    const void* ps = rad_planes_find(W->data, "scale", &es);
    if (!pc || !ps || !ec || !es) return RAD_E_DTYPE;
    if (ec->dtype != RAD_F8E4M3 || ec->block[0] != 1 || ec->block[1] != 1 ||
        ec->rows != N || ec->cols != K) return RAD_E_SHAPE;
    if (es->dtype != RAD_BF16 && es->dtype != RAD_F16 && es->dtype != RAD_F32)
        return RAD_E_UNSUPPORTED;
    if (es->block[1] != 128 || (es->block[0] != 1 && es->block[0] != 128))
        return RAD_E_UNSUPPORTED;
    const int64_t srows = (N + es->block[0] - 1) / es->block[0];
    const int64_t scols = (K + 127) / 128;
    if (es->rows != srows || es->cols != scols) return RAD_E_SHAPE;
    if (ec->row_bytes != K) return RAD_E_STRIDE;          /* dense, one byte each */
    if ((int64_t)h->bytes < es->offset + es->rows * es->row_bytes) return RAD_E_SHAPE;

    const int m_total = (int)M;
    const int mt   = m_total >= 8 ? 8 : m_total >= 4 ? 4 : m_total >= 2 ? 2 : 1;
    const int ygroups = (int)((m_total + mt - 1) / mt);
    const dim3 grid((unsigned)((N + 7) / 8), (unsigned)ygroups);

    const uint16_t* xp = (const uint16_t*)X->data;
    const uint8_t*  cp = (const uint8_t*)pc;
    const uint8_t*  sp = (const uint8_t*)ps;
    float*          yp = (float*)Y->data;
    const int scols_i = (int)scols, br_i = (int)es->block[0];
    const uint32_t sdt = es->dtype;

    switch (mt) {
        case 1: logits_e4m3_kernel<1><<<grid, 256, 0, (hipStream_t)s>>>(
                    xp, x_pitch, cp, ec->row_bytes, sp, scols_i, br_i, sdt,
                    yp, y_pitch, (int)N, (int)K, m_total, 0); break;
        case 2: logits_e4m3_kernel<2><<<grid, 256, 0, (hipStream_t)s>>>(
                    xp, x_pitch, cp, ec->row_bytes, sp, scols_i, br_i, sdt,
                    yp, y_pitch, (int)N, (int)K, m_total, 0); break;
        case 4: logits_e4m3_kernel<4><<<grid, 256, 0, (hipStream_t)s>>>(
                    xp, x_pitch, cp, ec->row_bytes, sp, scols_i, br_i, sdt,
                    yp, y_pitch, (int)N, (int)K, m_total, 0); break;
        default: logits_e4m3_kernel<8><<<grid, 256, 0, (hipStream_t)s>>>(
                    xp, x_pitch, cp, ec->row_bytes, sp, scols_i, br_i, sdt,
                    yp, y_pitch, (int)N, (int)K, m_total, 0); break;
    }
    const hipError_t le = hipGetLastError();
    if (le != hipSuccess) fprintf(stderr, "libr3d: logits_e4m3 launch FAILED (%s)\n",
                                  hipGetErrorString(le));
    return le == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------- layout hooks */

/* Whether `enc` and its selection are codes and one scale, untransformed -- the
 * ref's enc_scaled_rows rule, narrowed to e4m3 codes (this kernel does not read
 * int8 heads; refusing is honest, int8 heads then resolve to the host). */
extern "C" int r3d_layout_head(const RadParam*, int, int operand, const RadEncoding* enc,
                               const int* sel, const RadTensor* planes, int n, RadLayout* out) {
    if (n < 2 || operand != 1) return RAD_E_UNSUPPORTED;
    if (!enc || !sel || n != 2) return RAD_E_UNSUPPORTED;
    for (int i = 0; i < 2; ++i)
        if (sel[i] < 0 || sel[i] >= enc->n_planes) return RAD_E_DTYPE;
    if (enc->n_planes != 2 || enc->transform[0]) return RAD_E_DTYPE;
    if (strncmp(enc->plane[sel[0]].role, "codes", RAD_ENC_STR) != 0 ||
        strncmp(enc->plane[sel[1]].role, "scale", RAD_ENC_STR) != 0)
        return RAD_E_DTYPE;
    const RadEncPlane& c = enc->plane[sel[0]];
    const RadEncPlane& sl = enc->plane[sel[1]];
    if (c.kind != RAD_PLANE_TILED || c.block[0] != 1 || c.block[1] != 1 ||
        c.dtype != RAD_F8E4M3)
        return RAD_E_DTYPE;
    if (sl.kind != RAD_PLANE_TILED ||
        (sl.dtype != RAD_BF16 && sl.dtype != RAD_F16 && sl.dtype != RAD_F32))
        return RAD_E_DTYPE;
    /* The scheme gate: ref accepts affine only for this shape. */
    if (strncmp(enc->scheme, "affine", RAD_ENC_STR) != 0) return RAD_E_DTYPE;
    return rad_planes_layout(enc, sel, planes, n, out);
}

extern "C" int r3d_relayout_head(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                 const int* sel, const RadTensor* planes, int n_planes, void* dst,
                                 int64_t dst_bytes) {
    RadLayout lay;
    const int st = r3d_layout_head(p, n_p, operand, enc, sel, planes, n_planes, &lay);
    if (st != RAD_OK) return st;
    return rad_planes_relayout(enc, sel, planes, n_planes, dst, dst_bytes);
}

extern "C" int r3d_unrelayout_head(const RadParam* p, int n_p, int operand,
                                   const RadEncoding* enc, const int* sel, const void* src,
                                   int64_t src_bytes, const RadTensor* planes, int n_planes) {
    RadLayout lay;
    const int st = r3d_layout_head(p, n_p, operand, enc, sel, planes, n_planes, &lay);
    if (st != RAD_OK) return st;
    return rad_planes_unrelayout(enc, sel, src, src_bytes, planes, n_planes);
}

/* No opd_shape here: libref ships ref_logits_gemm_shape for the whole
 * conventional op, and kbench sizes the buffers with it (the same route the
 * mxfp4 GEMM row takes). */
