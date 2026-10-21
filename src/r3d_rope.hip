/* r3d_rope.hip -- libr3d's `rope` on RDNA3: rotary embedding in place over a fused qkv row.

 * The arithmetic is libref's (ref_attn.cpp ref_rope / rope_freqs / rope_head / rope_pos,
 * Apache-2.0, (c) radiance authors -- ported with attribution), with the rounding in the same
 * places:
 *   * the frequency ladder is the oracle's width-for-width: 1.0f / (float)pow((double)theta,
 *     (double)((float)(2i) / (float)rot)) -- the exponent is an f32 divide widened to double,
 *     the result narrowed to f32 and inverted there, exactly like ref_pow's double path;
 *   * the angle is pos * inv in f32 and the pair rotates with the accurate sincosf in f32 --
 *     the same choice libr4d's row documents ("fp32 angles through the accurate sincosf"):
 *     sincosf vs (float)cos(double) differs by ~1 f32-ULP, invisible under the bf16 store;
 *   * the two products and the difference are separate f32 steps (-ffp-contract=off) and the
 *     store is the single RTNE narrowing.
 *
 * Scope: the neox and gptj pairings over [M] positions, plus `imrope` -- Qwen3-VL's interleaved
 * M-RoPE over NeoX pairs (the 27B GDN trunk's attention) -- over [3, M] component-major
 * positions and `sections`. `mrope`/`axial`, a non-rotary ladder, yarn and llama3 frequency
 * scaling are refused at launch, not silently approximated -- and the row's constraints keep the
 * selector from asking. With rank-1 positions the reference calls rope_freqs with no sections at
 * all, so imrope there is exactly neox and rides the single-component kernel (ref_attn.cpp:
 * "mrope and imrope over a text batch are neox exactly").
 *
 * Layout: one block per token; the frequency ladder is built once per block in LDS (the heads
 * within a token share it), then threads stride the (head, pair) work of the Q and K heads.
 * The V heads that follow are untouched.
 */

#include <cstring>
#include "r3d_dump.h"

#include <hip/hip_runtime.h>

#include "rad_abi.h"
#include "rad_plugin.h"

namespace {

constexpr uint32_t kThreads = 256;
constexpr int kGptj = 1;   /* ref's kModes index for gptj; 0 is neox */

__device__ __forceinline__ float r3d_bf16_to_f32(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}
__device__ __forceinline__ uint16_t r3d_f32_to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* One token row: build the ladder in shared, then rotate `heads` = n_head + n_head_kv heads,
 * one (head, pair) item per striding thread. `Tok` is the positions dtype (the schema's i32;
 * i64 and f32 -- a fractional coordinate's own dtype -- are accepted for cold callers). */
template <typename Tok>
__global__ __launch_bounds__(kThreads) void rope_kernel(
    uint16_t* __restrict__ qkv, const Tok* __restrict__ pos,
    int64_t row_pitch, int64_t pos_stride, int64_t ds,
    uint32_t head_dim, uint32_t half, uint32_t rot, uint32_t heads, int mode,
    float theta, float div) {
    extern __shared__ float inv[];
    const int64_t m = blockIdx.x;

    /* The ladder, ref_rope_freqs' expression at the oracle's widths. The row refuses any
     * ladder but `rotary`, whose denominator is rot itself. */
    for (uint32_t i = threadIdx.x; i < half; i += kThreads)
        inv[i] = 1.0f / (float)pow((double)theta, (double)((float)(2u * i) / (float)rot));
    __syncthreads();

    /* rope_pos at the oracle's offset: ref reads offlin(pos, m), which is m * the position
     * tensor's element stride -- a view into a packed buffer is expressible and recorded. */
    const float p = (float)(double)pos[m * pos_stride] / div;

    const uint16_t* const row = qkv + m * row_pitch;
    const uint32_t work = heads * half;
    for (uint32_t w = threadIdx.x; w < work; w += kThreads) {
        const uint32_t h = w / half, i = w - h * half;
        const uint32_t i0 = (mode == kGptj) ? 2u * i : i;
        const uint32_t i1 = (mode == kGptj) ? 2u * i + 1u : i + half;
        /* The head axis is part of the address: ref reads x[i0 + h*head_dim]. Omitting it
         * makes every head write the first head's elements -- overlapping read-modify-writes
         * that compound rotations. */
        uint16_t* __restrict__ xr = (uint16_t*)row + (int64_t)h * head_dim;
        const float ang = p * inv[i];
        float c, s;
        sincosf(ang, &s, &c);                    /* magnitude is 1 outside yarn, which is refused */
        const float x0 = r3d_bf16_to_f32(xr[i0 * ds]);
        const float x1 = r3d_bf16_to_f32(xr[i1 * ds]);
        xr[i0 * ds] = r3d_f32_to_bf16(x0 * c - x1 * s);
        xr[i1 * ds] = r3d_f32_to_bf16(x0 * s + x1 * c);
    }
}

/* The imrope (Qwen3-VL interleaved M-RoPE) arm: [3, M] component-major positions. One block per
 * token, same ladder in LDS as the single-component kernel; each (head, pair) item picks its
 * position component by ref_rope's rule -- component 1 where i % 3 == 1 && i < 3*s1, component 2
 * where i % 3 == 2 && i < 3*s2, component 0 everywhere else -- and reads it at
 * `comp * pcs + m * pms`, exactly ref_rope's per-component rope_pos offset. NeoX pairing, one
 * ladder: everything else about the rotation is the kernel above. */
template <typename Tok>
__global__ __launch_bounds__(kThreads) void rope_imrope_kernel(
    uint16_t* __restrict__ qkv, const Tok* __restrict__ pos,
    int64_t row_pitch, int64_t pcs, int64_t pms, int64_t ds,
    uint32_t head_dim, uint32_t half, uint32_t rot, uint32_t heads,
    uint32_t s1, uint32_t s2, float theta, float div) {
    extern __shared__ float inv[];
    const int64_t m = blockIdx.x;

    for (uint32_t i = threadIdx.x; i < half; i += kThreads)
        inv[i] = 1.0f / (float)pow((double)theta, (double)((float)(2u * i) / (float)rot));
    __syncthreads();

    const uint16_t* const row = qkv + m * row_pitch;
    const uint32_t work = heads * half;
    for (uint32_t w = threadIdx.x; w < work; w += kThreads) {
        const uint32_t h = w / half, i = w - h * half;
        const uint32_t c = (i % 3u == 1u && i < 3u * s1) ? 1u
                       : (i % 3u == 2u && i < 3u * s2) ? 2u : 0u;
        const float p = (float)(double)pos[(int64_t)c * pcs + m * pms] / div;
        uint16_t* __restrict__ xr = (uint16_t*)row + (int64_t)h * head_dim;
        const float ang = p * inv[i];
        float cn, sn;
        sincosf(ang, &sn, &cn);
        const float x0 = r3d_bf16_to_f32(xr[i * ds]);
        const float x1 = r3d_bf16_to_f32(xr[(i + half) * ds]);
        xr[i * ds] = r3d_f32_to_bf16(x0 * cn - x1 * sn);
        xr[(i + half) * ds] = r3d_f32_to_bf16(x0 * sn + x1 * cn);
    }
}

} /* anonymous namespace */

/* Launch entry. Operands positional in the rope schema: qkv (INOUT), positions (IN). */
static int r3d_rope_impl(const RadArgs*, RadStream);
extern "C" int r3d_rad_rope(const RadArgs* a, RadStream s)
    R3D_DUMP_WRAP("rope_bf16", r3d_rope_impl)
static int r3d_rope_impl(const RadArgs* a, RadStream s) {
    const RadTensor* qkv = rad_arg_in(a, 0);   /* INOUT role: read as descriptor, written via data */
    const RadTensor* pos = rad_arg_in(a, 1);
    if (!qkv || !pos) return RAD_E_INVAL;
    if (qkv->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (pos->dtype != RAD_I32 && pos->dtype != RAD_I64 && pos->dtype != RAD_F32)
        return RAD_E_DTYPE;

    long long head_dim = 0, n_head = 0, n_head_kv = 0, rot = 0;
    if (!rad_args_geti(a, "head_dim", &head_dim) || head_dim <= 0) return RAD_E_SHAPE;
    if (!rad_args_geti(a, "n_head", &n_head) || n_head <= 0) return RAD_E_SHAPE;
    if (!rad_args_geti(a, "n_head_kv", &n_head_kv) || n_head_kv < 0) return RAD_E_SHAPE;
    rot = rad_args_geti_or(a, "rotary_dim", head_dim);
    if (rot <= 0 || rot > head_dim || (rot & 1)) return RAD_E_SHAPE;

    /* Modes, ref's kModes order; the row serves neox (0), gptj (1) and imrope (3). mrope (2)
     * and axial (4) resolve to no device row on this card -- an honest fallback, not an
     * approximation. */
    const char* ms = rad_args_gets(a, "mode");
    if (!ms) return RAD_E_INVAL;
    const int mode = !std::strcmp(ms, "neox") ? 0 : !std::strcmp(ms, "gptj") ? 1
                   : !std::strcmp(ms, "imrope") ? 3 : -1;
    if (mode < 0) return RAD_E_UNSUPPORTED;
    const bool multi = pos->rank >= 2;
    if (multi && mode != 3) return RAD_E_UNSUPPORTED;    /* the multi arm is imrope only */

    /* Sections, ref's rope_sections: a whitespace-free strtoll run, a non-positive part is a
     * shape error. The single-component arm keeps its shipped refusal of any sections string
     * (the reference ignores them at rank 1, but a stray ladder request should not reach the
     * kernel by loosening an arm that is gated in the field); imrope needs at least three. */
    int64_t sec[8] = {};
    int n_sec = 0;
    {
        const char* v = rad_args_gets(a, "sections");
        while (v && *v && n_sec < 8) {
            char* end = nullptr;
            const long long x = std::strtoll(v, &end, 10);
            if (end == v) break;
            if (x <= 0) return RAD_E_SHAPE;
            sec[n_sec++] = x;
            v = end;
        }
    }
    if (n_sec > 0 && mode != 3) return RAD_E_UNSUPPORTED;
    if (multi && n_sec < 3) return RAD_E_UNSUPPORTED;

    /* Refusals, each a shape the oracle computes differently than this kernel: the oracle
     * ignores sections and yarn for single-component neox/gptj too, but a mistyped ladder or
     * a scaling request reaching this row would compute silently different angles. */
    if (rad_args_gets(a, "ladder")) return RAD_E_UNSUPPORTED;
    if (rad_args_getf_or(a, "yarn_factor", 0.0) != 0.0 ||
        rad_args_getf_or(a, "llama3_factor", 0.0) != 0.0) return RAD_E_UNSUPPORTED;

    const float theta = (float)rad_args_getf_or(a, "theta", 10000.0);
    const float scale = (float)rad_args_getf_or(a, "scale", 1.0);
    const float div = scale != 0.0f ? scale : 1.0f;

    /* ref's offlin(pos, m) for a rank-1 position tensor: m * its element stride. At rank 2
     * ref_rope walks c * stride[0] + m * stride[1] and takes M from shape[1] (the components
     * are not tokens). */
    int64_t pos_stride = 1, pcs = 0, pms = 0, M;
    if (multi) {
        if (pos->rank != 2) return RAD_E_SHAPE;
        if (pos->shape[0] < 3 || pos->shape[0] > 8) return RAD_E_SHAPE;   /* ref's n_comp guard */
        pcs = pos->stride[0];
        pms = pos->stride[1];
        M = pos->shape[1];
    } else {
        pos_stride = pos->rank >= 1 ? pos->stride[pos->rank - 1] : 1;
        M = rad_tensor_numel(pos);
    }
    const int64_t stride_row = (n_head + 2 * n_head_kv) * head_dim;
    if (M <= 0 || M > 0x7fffffffLL) return RAD_E_SHAPE;
    if (rad_tensor_numel(qkv) < M * stride_row) return RAD_E_SHAPE;
    if (qkv->rank >= 2 && qkv->stride[qkv->rank - 1] != 1) return RAD_E_STRIDE;

    /* The token pitch, mirroring ref_gemm.h's row_stride(qkv, stride_row) exactly: fold the
     * trailing dimensions until they span the row, and the stride one axis out carries it.
     * For a rank-3 [M, heads, head_dim] qkv -- what the q-only (n_head_kv=0) geometries
     * present -- that is stride[0]; a bare stride[rank-2] there would read every token at
     * the head stride. */
    int64_t row_pitch;
    if (qkv->rank < 2) {
        row_pitch = stride_row;
    } else {
        int64_t span = 1;
        int hit = -1;
        for (int i = (int)qkv->rank - 1; i >= 0; --i) {
            span *= qkv->shape[i];
            if (span == stride_row) { hit = i; break; }
            if (span > stride_row) break;
        }
        row_pitch = hit >= 0 ? (hit > 0 ? qkv->stride[hit - 1] : stride_row)
                             : qkv->stride[qkv->rank - 2];   /* shapes disagree: ref's fallback */
    }
    const int64_t ds = 1;   /* element stride, enforced unit above; ref reads laststride(qkv) */

    const uint32_t half = (uint32_t)(rot / 2);
    const uint32_t heads = (uint32_t)(n_head + n_head_kv);
    const size_t shmem = (size_t)half * sizeof(float);

    hipStream_t stream = (hipStream_t)(uintptr_t)s;
    if (multi) {   /* imrope: the guards above proved mode 3, rank 2, n_sec >= 3 */
        const uint32_t s1 = (uint32_t)sec[1], s2 = (uint32_t)sec[2];
        if (pos->dtype == RAD_I32)
            hipLaunchKernelGGL((rope_imrope_kernel<int32_t>), dim3((uint32_t)M), dim3(kThreads),
                               shmem, stream, (uint16_t*)qkv->data, (const int32_t*)pos->data,
                               row_pitch, pcs, pms, ds, (uint32_t)head_dim, half,
                               (uint32_t)rot, heads, s1, s2, theta, div);
        else if (pos->dtype == RAD_I64)
            hipLaunchKernelGGL((rope_imrope_kernel<int64_t>), dim3((uint32_t)M), dim3(kThreads),
                               shmem, stream, (uint16_t*)qkv->data, (const int64_t*)pos->data,
                               row_pitch, pcs, pms, ds, (uint32_t)head_dim, half,
                               (uint32_t)rot, heads, s1, s2, theta, div);
        else
            hipLaunchKernelGGL((rope_imrope_kernel<float>), dim3((uint32_t)M), dim3(kThreads),
                               shmem, stream, (uint16_t*)qkv->data, (const float*)pos->data,
                               row_pitch, pcs, pms, ds, (uint32_t)head_dim, half,
                               (uint32_t)rot, heads, s1, s2, theta, div);
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
    /* rank-1 imrope reaches the single-component kernel: the reference builds its frequencies
     * with no sections at rank 1, so every component is the token's one position. */
    const int pair_mode = mode == 1 ? kGptj : 0;
    if (pos->dtype == RAD_I32)
        hipLaunchKernelGGL((rope_kernel<int32_t>), dim3((uint32_t)M), dim3(kThreads),
                           shmem, stream, (uint16_t*)qkv->data, (const int32_t*)pos->data,
                           row_pitch, pos_stride, ds, (uint32_t)head_dim, half, (uint32_t)rot, heads,
                           pair_mode, theta, div);
    else if (pos->dtype == RAD_I64)
        hipLaunchKernelGGL((rope_kernel<int64_t>), dim3((uint32_t)M), dim3(kThreads),
                           shmem, stream, (uint16_t*)qkv->data, (const int64_t*)pos->data,
                           row_pitch, pos_stride, ds, (uint32_t)head_dim, half, (uint32_t)rot, heads,
                           pair_mode, theta, div);
    else
        hipLaunchKernelGGL((rope_kernel<float>), dim3((uint32_t)M), dim3(kThreads),
                           shmem, stream, (uint16_t*)qkv->data, (const float*)pos->data,
                           row_pitch, pos_stride, ds, (uint32_t)head_dim, half, (uint32_t)rot, heads,
                           pair_mode, theta, div);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ rope_table */
/* The cos/sin plane a fused prologue reads instead of theta: row `pos[m]` of a
 * [rows, rot] table, cos in the first half of the row and sin in the second. Rows
 * outside [0, rows) write nothing -- ref `continue`s on them, and a guarded-allocation
 * sweep sees exactly that.
 *
 * Same ladder and the same sincosf choice as the rope kernel above, so a fused prologue
 * reading this table and the in-place rope agree at the last operation of the ladder.
 * The delta ref_cos(double) would make is ~1 f32-ULP: at an f32 store it surfaces as
 * ~1e-7 rel_l2, three-plus orders under any recorded tolerance. */
template <typename Tok>
__global__ __launch_bounds__(kThreads) void rope_table_kernel(
    const Tok* __restrict__ pos, uint8_t* __restrict__ cs, uint32_t dt_cs,
    int64_t cs_pitch_bytes, int64_t pos_stride,
    uint32_t rot, uint32_t half, int64_t table_rows, float theta, float div) {
    __shared__ float inv[512];                    /* the row's ladder; rot <= 1024 refused at entry */
    const int64_t pp = (int64_t)pos[blockIdx.x * pos_stride];
    if (pp < 0 || pp >= table_rows) return;
    for (uint32_t i = threadIdx.x; i < half; i += kThreads)
        inv[i] = 1.0f / (float)pow((double)theta, (double)((float)(2u * i) / (float)rot));
    __syncthreads();
    const float p = (float)pp / div;
    uint8_t* __restrict__ row = cs + pp * cs_pitch_bytes;
    for (uint32_t i = threadIdx.x; i < half; i += kThreads) {
        const float ang = p * inv[i];
        float s, c;
        sincosf(ang, &s, &c);
        if (dt_cs == RAD_F32) {
            ((float*)row)[i] = c;
            ((float*)row)[half + i] = s;
        } else {
            uint16_t* h = (uint16_t*)row;
            h[i] = r3d_f32_to_bf16(c);
            h[half + i] = r3d_f32_to_bf16(s);
        }
    }
}

static int r3d_rope_table_impl(const RadArgs*, RadStream);
extern "C" int r3d_rad_rope_table(const RadArgs* a, RadStream s)
    R3D_DUMP_WRAP("rope_table_f32", r3d_rope_table_impl)
static int r3d_rope_table_impl(const RadArgs* a, RadStream s) {
    const RadTensor* pos = rad_arg_in(a, 0);
    const RadTensor* cs  = rad_arg_in(a, 1);
    if (!pos || !cs) return RAD_E_INVAL;
    if (pos->dtype != RAD_I32 && pos->dtype != RAD_I64) return RAD_E_DTYPE;
    if (cs->dtype != RAD_F32 && cs->dtype != RAD_BF16) return RAD_E_DTYPE;

    long long rot_ll = 0;
    if (!rad_args_geti(a, "rot", &rot_ll)) return RAD_E_INVAL;
    if (rot_ll < 2 || (rot_ll & 1) || rot_ll > 1024) return RAD_E_SHAPE;   /* __shared__ bound */
    const int64_t rot = rot_ll;

    /* Refusals for anything whose angles this kernel would not reproduce: YaRN blends
     * the ladder and scales the plane, and a `head` ladder denominates by head_dim.
     * A `rotary` ladder is the no-ladder case for ref, so it is accepted. */
    const char* lad = rad_args_gets(a, "ladder");
    if (lad && std::strcmp(lad, "rotary")) return RAD_E_UNSUPPORTED;
    if (rad_args_getf_or(a, "yarn_factor", 0.0) != 0.0 ||
        rad_args_getf_or(a, "yarn_low", 0.0) != 0.0 ||
        rad_args_getf_or(a, "yarn_high", 0.0) != 0.0 ||
        rad_args_getf_or(a, "yarn_magnitude", 1.0) != 1.0) return RAD_E_UNSUPPORTED;

    const float theta = (float)rad_args_getf_or(a, "theta", 10000.0);
    const float scale = (float)rad_args_getf_or(a, "scale", 1.0);
    const float div = scale != 0.0f ? scale : 1.0f;

    if (pos->rank >= 2) return RAD_E_UNSUPPORTED;    /* multi-component positions */
    const int64_t pos_stride = pos->rank >= 1 ? pos->stride[pos->rank - 1] : 1;
    const int64_t M = rad_tensor_numel(pos);
    const int64_t numel_cs = rad_tensor_numel(cs);
    if (M <= 0 || M > 0x7fffffffLL) return RAD_E_SHAPE;
    if (numel_cs % rot) return RAD_E_SHAPE;
    const int64_t table_rows = numel_cs / rot;

    /* The table's row pitch: a [rows, rot] rank-2 with unit last stride, or a bare
     * rank-1 walked as rows of rot (ref's offlin is the identity on it). */
    int64_t cs_pitch;
    if (cs->rank <= 1) cs_pitch = rot;
    else if (cs->rank == 2 && cs->stride[1] == 1) cs_pitch = cs->stride[0];
    else return RAD_E_STRIDE;

    hipStream_t stream = (hipStream_t)(uintptr_t)s;
    const uint32_t half = (uint32_t)(rot / 2);
    if (pos->dtype == RAD_I32)
        hipLaunchKernelGGL((rope_table_kernel<int32_t>), dim3((uint32_t)M), dim3(kThreads),
                           0, stream, (const int32_t*)pos->data, (uint8_t*)cs->data, cs->dtype,
                           rad_dtype_bytes(cs->dtype, cs_pitch), pos_stride,
                           (uint32_t)rot, half, table_rows, theta, div);
    else
        hipLaunchKernelGGL((rope_table_kernel<int64_t>), dim3((uint32_t)M), dim3(kThreads),
                           0, stream, (const int64_t*)pos->data, (uint8_t*)cs->data, cs->dtype,
                           rad_dtype_bytes(cs->dtype, cs_pitch), pos_stride,
                           (uint32_t)rot, half, table_rows, theta, div);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
