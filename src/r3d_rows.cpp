/* r3d_rows.cpp -- libr3d's declaration: plugin identity and the kernel row table.
 *
 * Pure host C++ including only the ABI, so it compiles without ROCm, exactly like libr4d's
 * r4d_rows.cpp. This skeleton exports the full kernel-plugin contract with an empty table:
 * the loader takes it (ABI matches), finds no device rows for gfx11, and selection behaves as
 * if libr3d were absent. Real rows land here as kernels are implemented; docs/COVERAGE.md
 * tracks the order.
 */

#include "rad_abi.h"

extern "C" {

uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }

const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo info = {
        RAD_PLUGIN_KERNEL,
        "libr3d",
        "0.1.0",
        "radiance kernel library for AMD RDNA3 (gfx11)",
        "gfx1100 gfx1101 gfx1102 gfx1103 gfx11-generic",
    };
    return &info;
}

/* Kernel rows and their op schemas. Every op a plugin implements must have its schema
 * declared by some *loaded kernel plugin* (loader.cpp stages rad_kernel_schema_count/at per
 * plugin; libref is a kbench oracle, not a search-path plugin, so it fixes nothing). On RDNA3
 * libr3d is that declarer, and a kernel whose op has no declared schema is an error:
 *   "implements op 'rmsnorm', which no loaded plugin declares a schema for"
 * The schema below is copied verbatim from libr4d's kSchemas entry, so where libr4d is also
 * present (an RDNA4 box or a mixed rig) the loader's schema_same() check passes and first
 * declarer wins with no refusal. Any future op libr3d implements must likewise arrive with
 * its schema declared here, identical to libr4d's where one exists. */

/* rmsnorm: verbatim libr4d (r4d_rows.cpp pRmsNorm/oRmsNorm). `wadd` is GemmaRMSNorm's `1 +`;
 * `mult` scales the row before its one rounding. `w` optional: absent, the gain is one. */
static const RadParamSpec pRmsNorm[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED },
    { "n",     RAD_P_INT, RAD_REQUIRED },
    { "eps",   RAD_P_F64, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
    { "wadd",  RAD_P_F64, RAD_OPTIONAL },
    { "mult",  RAD_P_F64, RAD_OPTIONAL },
};
static const RadOperandSpec oRmsNorm[] = {
    { "x", RAD_OPD_IN,     0 },
    { "w", RAD_OPD_WEIGHT, 1 },
    { "y", RAD_OPD_OUT,    0 },
};

static const RadOpSchema kSchemas[] = {
    { "rmsnorm", pRmsNorm, (int)(sizeof(pRmsNorm)/sizeof(pRmsNorm[0])),
      oRmsNorm,  (int)(sizeof(oRmsNorm)/sizeof(oRmsNorm[0])),
      "y = x * rsqrt(mean(x^2) + eps) * (w + wadd). Declared verbatim from libr4d so both "
      "plugins agree wherever they co-load." },
};

/* The rmsnorm row. Field order is RadKernelInfo's positional form (rad_abi.h): after the prose,
 * constraints and (absent) tunables come init, fini, launch, scratch, layout, relayout, and
 * opd_shape -- null because a cold-calling tool takes operand shapes from libref's for the op.
 * `replaces` stays zero-initialised: this row is the unfused op, not a fusion. */

static const RadConstraint cBf16Row[] = {
    { "dtype", RAD_C_IN, 0, "bf16" },
};

/* libr3d is the only device library on RDNA3, so no in-plugin arbitration; the value only
 * needs to sit above a future fallback row. */
enum { R3D_PRIO_ONLY = 1000 };

extern "C" int r3d_rad_rmsnorm(const RadArgs*, RadStream);

static const RadKernelInfo kRows[] = {

{ "rmsnorm_bf16", "rmsnorm", "norm",
  "root-mean-square norm with a gain, two passes over the row",
  "any n; the sum of squares accumulates in f32 and only the store narrows",
  "bf16 x and y, bf16 or f32 gain, fp32 reduction",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  cBf16Row, 1, nullptr, 0, nullptr,
  nullptr, nullptr, r3d_rad_rmsnorm },

};

int rad_kernel_schema_count(void) { return (int)(sizeof(kSchemas) / sizeof(kSchemas[0])); }
const RadOpSchema* rad_kernel_schema_at(int i) {
    return (i >= 0 && i < rad_kernel_schema_count()) ? &kSchemas[i] : nullptr;
}

int rad_kernel_count(void) { return (int)(sizeof(kRows) / sizeof(kRows[0])); }
const RadKernelInfo* rad_kernel_at(int i) {
    return (i >= 0 && i < rad_kernel_count()) ? &kRows[i] : nullptr;
}

} /* extern "C" */
