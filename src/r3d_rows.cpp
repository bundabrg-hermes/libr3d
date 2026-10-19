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

/* Kernel rows and their op schemas. Empty for now; every conventional op libr3d implements
 * gets a RadKernelInfo row here and a RadOpSchema mirroring libref's, whose schema wins if
 * libref declares the same op first (the loader refuses disagreeing schemas). */

int rad_kernel_schema_count(void) { return 0; }
const RadOpSchema* rad_kernel_schema_at(int) { return nullptr; }

int rad_kernel_count(void) { return 0; }
const RadKernelInfo* rad_kernel_at(int) { return nullptr; }

} /* extern "C" */
