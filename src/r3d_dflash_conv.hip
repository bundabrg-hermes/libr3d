/* r3d_dflash_conv.hip -- dflash_conv on RDNA3: DFlash2's grouped dynamic depthwise
 * convolution over the [T, H] rows of x.
 *
 * DRAFT for compilation/gating by the owning session -- not yet in CMakeLists, not yet a
 * registered row. The row-table entry, schemas, and registration filter are spelled out in
 * the comment block at the bottom of this file and in docs (scratch/drafter_rows_design.md).
 *
 * The oracle is libref ref_qsa.cpp ref_dflash_conv (Apache-2.0, (c) radiance authors --
 * ported with attribution). Per output element out[t][h], in ASCENDING tap order j:
 *
 *     if (j > 0 && t % block_size < j)  skip tap j            (block-start mask)
 *     c    = f32(base[j][h]) + f32(delta[t][j][h / group])    (one f32 add rounding)
 *     term = c * f32(x[t - j][h])                             (one f32 mul rounding)
 *     acc  = (j == 0) ? term : acc + term                     (ascending f32 adds)
 *
 * then ONE narrowing to out's dtype (RTNE). No FMA anywhere: the coefficient add and the
 * product are separately rounded -- "-ffp-contract=off" (plugin-wide in CMakeLists) is what
 * keeps the mul+add pair from fusing into a differently-rounded chain on device. With that
 * flag and bf16->f32 widening being exact, the kernel reproduces the oracle's every
 * rounding step bit-for-bit; like ref's own REF_PAR_FOR loop the work is embarrassingly
 * parallel over (t, h), each element a private chain, so no reduction order to match.
 *
 * The j == 0 special case is kept VERBATIM (`acc = term`, not `acc = 0 + term`): for
 * term = -0.0f the two differ (+0 vs -0) and a kbench fixture can carry a zero row.
 *
 * Shape scope (constraints, verbatim libr4d's cDflashConv so a drafter geometry records the
 * same band on both libraries): taps == 2, group_size == 16, hidden_size % 128, dtype bf16.
 * The launcher additionally mirrors every guard ref performs (rank 2 x/out with equal
 * shapes, H == NG * group_size, base numel, delta rank 2/3 extents, the x==out alias
 * refusal). Those entry checks are a BACKSTOP for what the constraint table already
 * excludes -- the engine selects at declare time, so a refusal at issue is an abort, not a
 * re-resolve (the issue #1 / R3D_NO_GDN lesson).
 *
 * delta arrives as a slice of the [T, 2, taps, NG] coefficient projection: rank 2
 * [T, >= taps * NG] read at its own row pitch (the shape fn spells [T, 2 * taps * NG]),
 * or rank 3 [T, taps, NG] with its own three strides. Both are read by passing the
 * strides through to the kernel, exactly as ref's d_rs / d_ts / d_gs do.
 *
 * R3D_NO_DFLASH=1 withdraws this row (and its dflash siblings) from the REGISTRATION
 * enumeration -- see the filter snippet below -- and the entry backstops the same env so a
 * forced launch declines instead of running.
 */
#include <hip/hip_runtime.h>
#include "r3d_dump.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rad_abi.h"
#include "rad_plugin.h"   /* rad_arg_in / rad_tensor_numel (host-side entry helpers) */

namespace {

constexpr uint32_t kThreads = 256;

/* Device mirrors of the ABI conversions, the r3d_more.hip pattern (the rad_plugin.h
 * helpers are host-only linkage under HIP's clang). */
__device__ __forceinline__ float r3d_bf16_to_f32(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}
__device__ __forceinline__ uint16_t r3d_f32_to_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* One thread per (t, h). The whole chain is register-resident: taps tops out at 2 under
 * the row's constraints, so this is a pure streaming pass -- read base[j][h] once per
 * (j, h) (L2-hot across t), delta[t][j][g], x[t-j][h]; write out[t][h]. */
__global__ void dflash_conv_kernel(const uint16_t* __restrict__ x,
                                   const uint16_t* __restrict__ dl,
                                   const uint16_t* __restrict__ bs,
                                   uint16_t* __restrict__ o,
                                   int64_t T, int64_t H,
                                   int64_t x_rs, int64_t o_rs,
                                   int64_t d_rs, int64_t d_ts,
                                   int64_t bs_rs,
                                   int taps, int group, int bsz) {
    const int64_t total = T * H;
    const int64_t i64 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i64 >= total) return;
    /* Q4 (REVIEW_QUEUE): the launcher caps total <= 0xffff*kThreads, so every index
     * below provably fits u32 -- the three i64 software divisions become u32, and the
     * per-tap t%bsz is hoisted to once per element (same value every tap). Addresses
     * stay int64; values are bit-identical integer identities. */
    const uint32_t i = (uint32_t)i64, H32 = (uint32_t)H;
    const uint32_t t = i / H32, h = i - t * H32;
    const uint32_t g = h / (uint32_t)group, mask = t % (uint32_t)bsz;

    float acc = 0.0f;
    for (int j = 0; j < taps; ++j) {
        if (j > 0 && mask < (uint32_t)j) continue;             /* the block-start mask */
        const float c = r3d_bf16_to_f32(bs[j * bs_rs + h]) +
                        r3d_bf16_to_f32(dl[t * d_rs + (int64_t)j * d_ts + g]);
        const float term = c * r3d_bf16_to_f32(x[(t - j) * x_rs + h]);
        acc = j == 0 ? term : acc + term;                      /* verbatim ref's rounding */
    }
    o[t * o_rs + h] = r3d_f32_to_bf16(acc);
}

/* Host helpers, the r3d_more.hip pattern. */
static inline bool dt_bf16(uint32_t dt) { return dt == RAD_BF16; }
static inline int64_t param_i(const RadArgs* a, const char* k, int64_t fb) {
    long long v;
    return rad_args_geti(a, k, &v) ? (int64_t)v : fb;
}
/* Row pitch of a rank-2 [rows, cols] operand read with rowoff's rule: trailing stride
 * must be unit; the row stride is stride[0] when shape[1] == cols, the flat r*cols
 * fallback otherwise (exactly what the reference's rowoff walks into for our scope). */
static inline int row2(const RadTensor* t, int64_t cols, int64_t* pitch) {
    if (!t || t->rank != 2 || t->stride[1] != 1) return -1;
    *pitch = (t->shape[1] == cols) ? t->stride[0] : cols;
    return 0;
}

/* R3D_DEBUG_DFLASH=1: name the operand that tripped a refuse. The engine aborts the step
 * on a refusal at issue (declare-time selection is final), so the entry never re-resolves
 * and the abort line alone does not say WHICH operand's view deviated (serve 2026-10-24,
 * the drafter's first issue at T=8): rank/shape/stride of all four, every reject. */
void dbg_op(const char* tag, const RadTensor* x, const RadTensor* dl,
            const RadTensor* bs, const RadTensor* o) {
    if (!getenv("R3D_DEBUG_DFLASH")) return;
    const RadTensor* ts[4] = { x, dl, bs, o };
    static const char* nm[4] = { "x", "delta", "base", "out" };
    fprintf(stderr, "libr3d: dflash_conv refuse(%s):", tag);
    for (int i = 0; i < 4; ++i) {
        const RadTensor* t = ts[i];
        if (!t) { fprintf(stderr, " %s=(null)", nm[i]); continue; }
        fprintf(stderr, " %s[rank=%lld shape=", nm[i], (long long)t->rank);
        for (int d = 0; d < t->rank && d < 4; ++d)
            fprintf(stderr, "%s%lld", d ? "x" : "", (long long)t->shape[d]);
        fprintf(stderr, " stride=");
        for (int d = 0; d < t->rank && d < 4; ++d)
            fprintf(stderr, "%s%lld", d ? "/" : "", (long long)t->stride[d]);
        fprintf(stderr, "]");
    }
    fprintf(stderr, "\n");
}

} /* anonymous namespace */

/* ------------------------------------------------------------------ entry */
static int r3d_dflash_conv_impl(const RadArgs*, RadStream);
extern "C" int r3d_rad_dflash_conv(const RadArgs* a, RadStream s)
    R3D_DUMP_WRAP("dflash_conv", r3d_dflash_conv_impl)
static int r3d_dflash_conv_impl(const RadArgs* a, RadStream s) {
    /* Registration-time the row is withdrawn under this env (see the filter below); the
     * entry backstop covers a hand-forced launch. Never refuse a shape the constraints
     * accept -- that path is an abort, and the registration filter is the lever. */
    if (getenv("R3D_NO_DFLASH")) return RAD_E_UNSUPPORTED;

    /* ref: have(a, 4); operands x, delta, base, out. */
    if (a->n_t < 4) return RAD_E_INVAL;
    const RadTensor* x  = rad_arg_in(a, 0);
    const RadTensor* dl = rad_arg_in(a, 1);
    const RadTensor* bs = rad_arg_in(a, 2);
    const RadTensor* o  = rad_arg_in(a, 3);
    if (!x || !dl || !bs || !o) return RAD_E_INVAL;
    if (x->data == o->data) return RAD_E_INVAL;          /* ref's alias refusal, verbatim */
    if (!dt_bf16(x->dtype) || !dt_bf16(dl->dtype) ||
        !dt_bf16(bs->dtype) || !dt_bf16(o->dtype)) return RAD_E_DTYPE;

    const int64_t taps   = param_i(a, "taps", 0);
    const int64_t group  = param_i(a, "group_size", 0);
    const int64_t NG     = param_i(a, "NG", 0);
    const int64_t bsz    = param_i(a, "block_size", 0);
    if (taps <= 0 || group <= 0 || NG <= 0 || bsz <= 0) return RAD_E_INVAL;
    if (x->rank != 2 || o->rank != 2) { dbg_op("rank-xo", x, dl, bs, o); return RAD_E_SHAPE; }
    const int64_t T = x->shape[0], H = x->shape[1];
    if (T <= 0 || o->shape[0] != T || o->shape[1] != H || H != NG * group) {
        dbg_op("shape-xo", x, dl, bs, o); return RAD_E_SHAPE;
    }
    if (rad_tensor_numel(bs) < taps * H) { dbg_op("numel-bs", x, dl, bs, o); return RAD_E_SHAPE; }

    /* delta: rank 3 [T, >=taps, >=NG] at its own strides, or rank 2 [T, >=taps*NG] at its
     * own row pitch (group stride unit), exactly ref's two spellings. */
    int64_t d_rs = 0, d_ts = 0;
    if (dl->rank == 3) {
        if (dl->shape[0] < T || dl->shape[1] < taps || dl->shape[2] < NG) {
            dbg_op("ext-dl3", x, dl, bs, o); return RAD_E_SHAPE;
        }
        if (dl->stride[2] != 1) { dbg_op("unit-dl3", x, dl, bs, o); return RAD_E_STRIDE; }
        d_rs = dl->stride[0]; d_ts = dl->stride[1];
    } else if (dl->rank == 2) {
        if (dl->shape[0] < T || dl->shape[1] < taps * NG) {
            dbg_op("ext-dl2", x, dl, bs, o); return RAD_E_SHAPE;
        }
        if (dl->stride[1] != 1) { dbg_op("unit-dl2", x, dl, bs, o); return RAD_E_STRIDE; }
        d_rs = dl->stride[0]; d_ts = NG;                   /* ref: d_ts = NG * d_gs, d_gs = 1 */
    } else { dbg_op("rank-dl", x, dl, bs, o); return RAD_E_SHAPE; }

    int64_t x_rs = 0, o_rs = 0, bs_rs = 0;
    if (row2(x,  H, &x_rs))  { dbg_op("row2-x",  x, dl, bs, o); return RAD_E_STRIDE; }
    if (row2(o,  H, &o_rs))  { dbg_op("row2-o",  x, dl, bs, o); return RAD_E_STRIDE; }
    /* base's runtime spelling is the weight unit's batched [1, taps, H] view (serve
     * 2026-10-24: rank 3 stride {taps*H, H, 1}); kbench's recipe spells it rank 2. Fold
     * a degenerate batch prefix to its rank-2 body and read the row pitch from its own
     * strides -- addressing identical, ref's base reads are row t at t * pitch. */
    RadTensor bs2 = *bs;
    if (bs->rank == 3 && bs->shape[0] == 1) {
        bs2.rank = 2; bs2.shape[0] = bs->shape[1]; bs2.shape[1] = bs->shape[2];
        bs2.stride[0] = bs->stride[1]; bs2.stride[1] = bs->stride[2];
    }
    if (row2(&bs2, H, &bs_rs)) { dbg_op("row2-bs", x, dl, bs, o); return RAD_E_STRIDE; }

    const int64_t total = T * H;
    if (total <= 0 || total > (int64_t)0xffff * kThreads) return RAD_E_SHAPE;
    const dim3 grid((unsigned)((total + kThreads - 1) / kThreads));
    dflash_conv_kernel<<<grid, dim3(kThreads), 0, (hipStream_t)(uintptr_t)s>>>(
        (const uint16_t*)x->data, (const uint16_t*)dl->data, (const uint16_t*)bs->data,
        (uint16_t*)o->data, T, H, x_rs, o_rs, d_rs, d_ts, bs_rs,
        (int)taps, (int)group, (int)bsz);
    const hipError_t le = hipGetLastError();
    if (le != hipSuccess) fprintf(stderr, "libr3d: dflash_conv launch FAILED (%s)\n",
                                  hipGetErrorName(le));
    else if (getenv("R3D_DEBUG_DFLASH")) {
        static bool once = false;
        if (!once) { once = true;
            fprintf(stderr, "libr3d: dflash_conv launched (T=%lld H=%lld taps=%lld g=%lld "
                            "NG=%lld bsz=%lld)\n", (long long)T, (long long)H,
                    (long long)taps, (long long)group, (long long)NG, (long long)bsz); }
    }
    return le == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* ============================================================================
 * Row-table entry -- paste into src/r3d_rows.cpp (nothing in THIS file declares
 * the row; r3d_rows.cpp is the single declaration TU). Field order is
 * RadKernelInfo's positional order; keep `replaces` last (ABI comment law).
 *
 * 1) extern, beside the other launcher externs:
 *
 * extern "C" int r3d_rad_dflash_conv(const RadArgs*, RadStream);
 *
 * 2) schemas, verbatim libr4d's pDflashConv/oDflashConv (r4d_rows.cpp), so the
 *    loader's schema_same() duplicate check passes wherever both plugins load:
 *
 * static const RadParamSpec pDflashConv[] = {
 *     { "T",           RAD_P_INT, RAD_REQUIRED },
 *     { "hidden_size", RAD_P_INT, RAD_REQUIRED },
 *     { "taps",        RAD_P_INT, RAD_REQUIRED },
 *     { "group_size",  RAD_P_INT, RAD_REQUIRED },
 *     { "NG",          RAD_P_INT, RAD_REQUIRED },
 *     { "block_size",  RAD_P_INT, RAD_REQUIRED },
 *     { "dtype",       RAD_P_STR, RAD_REQUIRED },
 * };
 * static const RadOperandSpec oDflashConv[] = {
 *     { "x",     RAD_OPD_IN,     0 },
 *     { "delta", RAD_OPD_IN,     0 },
 *     { "base",  RAD_OPD_WEIGHT, 0 },
 *     { "out",   RAD_OPD_OUT,    0 },
 * };
 *
 *    and inside kSchemas[]:
 *
 *     SCHEMA-style entry (match r3d_rows.cpp's brace style for RadOpSchema):
 *     { "dflash_conv", pDflashConv, (int)(sizeof(pDflashConv)/sizeof(pDflashConv[0])),
 *       oDflashConv, (int)(sizeof(oDflashConv)/sizeof(oDflashConv[0])),
 *       "DFlash2 grouped dynamic depthwise convolution, coefficient add and both taps in "
 *       "one pass." },
 *
 * 3) constraints, verbatim libr4d's cDflashConv (taps 2 / group 16 / hidden % 128 / bf16
 *    is the only drafter form any shipped checkpoint carries; matching libr4d's band means
 *    drafter fixtures recorded for one library reach the other):
 *
 * static const RadConstraint cDflashConv[] = {
 *     { "taps",        RAD_C_EQ,  2,   nullptr },
 *     { "group_size",  RAD_C_EQ,  16,  nullptr },
 *     { "hidden_size", RAD_C_DIV, 128, nullptr },
 *     { "dtype",       RAD_C_IN,  0,   "bf16" },
 * };
 *
 * 4) the row, appended to kRows[]:
 *
 * { "dflash_conv_t2_g16_bf16", "dflash_conv", "dflash",
 *   "DFlash2 grouped dynamic depthwise conv: coefficient add, both taps and the "
 *   "block-position mask in one pass, one thread per (t, h)",
 *   "taps 2, group_size 16, hidden_size divisible by 128; delta is a slice of the "
 *   "[T, 2, taps, NG] projection read at its own row pitch (rank 3 at its own strides "
 *   "also served); base and out at their own row pitches; out must not alias x",
 *   "bf16 x / delta / base / out, fp32 accumulate in the oracle's tap order -- "
 *   "bit-exact by construction under -ffp-contract=off",
 *   RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
 *   cDflashConv, 4, nullptr, 0, nullptr,
 *   nullptr, nullptr, r3d_rad_dflash_conv },
 *
 * 5) kill switch -- R3D_NO_DFLASH=1 withdraws the three drafter rows from the
 *    registration enumeration (a launch-time refusal is an abort, not a re-resolve;
 *    the R3D_NO_GDN pattern). Add beside r3d_gdn_hidden() and thread through both
 *    rad_kernel_count() and rad_kernel_at():
 *
 * static bool r3d_dflash_row_hidden(const RadKernelInfo& k) {
 *     static const bool no_df = getenv("R3D_NO_DFLASH") != nullptr;
 *     if (!no_df) return false;
 *     return !strcmp(k.op, "dflash_conv") || !strcmp(k.op, "dflash_select") ||
 *            !strcmp(k.op, "logit_rerank");
 * }
 *    ... and in rad_kernel_count()/rad_kernel_at(): add
 *    `&& !r3d_dflash_row_hidden(k)` to both loop predicates. The op SCHEMAS stay
 *    declared either way (the fp8-lever precedent: a declared schema with no row is
 *    legal; the reverse is the load error).
 *
 * 6) CMake: libr3d's CMakeLists lists HIP sources EXPLICITLY (no glob) -- append
 *    src/r3d_dflash_conv.hip to rad_add_plugin's HIP_SOURCES and add the per-unit
 *    correctness flags beside the others:
 *
 * set_source_files_properties(src/r3d_dflash_conv.hip PROPERTIES
 *   COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")
 * ========================================================================== */
