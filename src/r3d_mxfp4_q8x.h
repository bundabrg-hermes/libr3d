/* r3d_mxfp4_q8x.h -- shared constants for the mxfp4 decode dot4 arm (hipfire Phase B).
 *
 * Header-visible so tools/dot4_fuzz_mxfp4.hip can fuzz these against an INDEPENDENT
 * decoder (rad_fp4e2m1_to_f32 from the ABI header): the probe-era law "e2m1 codes 7/15
 * are NaN, never generated" was a table-vs-copy-of-itself artifact -- the probe's
 * generator never emitted 7/15 and its f64 model shared its table, so both were
 * blind to the fact that OCP e2m1 has NO NaN ("all sixteen codes are numbers, max
 * 6", rad_plugin.h:356). The shipped kE2M1 bf16 table decodes code 7 as +6.0, so a
 * repack that zeroed 7/15 silently drops every +-6.0 weight on real checkpoints.
 */
#ifndef R3D_MXFP4_Q8X_H
#define R3D_MXFP4_Q8X_H

#include <cstdint>
#include <cstring>

/* e2m1 code -> value x 2 as i8 (the doubling keeps +-0.5 integral in the byte plane;
 * the dot flush carries the matching 0.5). Derived from the ABI's own magnitude
 * table { 0, .5, 1, 1.5, 2, 3, 4, 6 } x 2 -- codes 7/15 ARE numbers (+-6 -> +-12).
 * Negatives are -positive, never an arithmetic stride. Consumers expand the macro
 * into their own `constexpr int8_t lut[16]` so no storage crosses a TU boundary. */
#define R3D_E2M1_X2  { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 }

/* e8m0 scale byte -> f32, the shipped mxfp4 GEMV's decode verbatim: 0xFF is NaN and
 * poisons through the flush like the oracle; 0x00 is the subnormal 2^-127 that the
 * bare bit trick would flatten to zero; every other byte is the exact biased pattern.
 * (The probe arm used the bare trick with a uniform-positive scale generator; the
 * integration pays the two compares so adversarial scale bytes cannot diverge.) */
__device__ __host__ inline float r3d_e8m0_sc(uint8_t x) {
    uint32_t u = 0x7fc00000u;
    if (x != 0xffu) u = (x == 0x00u) ? 0x00400000u : ((uint32_t)x << 23);
    float f;
    unsigned char* d = (unsigned char*)&f;
    const unsigned char* s = (const unsigned char*)&u;
    for (int i = 0; i < 4; ++i) d[i] = s[i];      /* memcpy-free: usable in device code */
    return f;
}

#endif /* R3D_MXFP4_Q8X_H */
