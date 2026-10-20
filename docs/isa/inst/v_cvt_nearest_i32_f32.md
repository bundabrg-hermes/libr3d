---
mnemonic: V_CVT_NEAREST_I32_F32
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "12"
pages: [277]
source: RDNA3 ISA PDF p.277-277
---

# V_CVT_NEAREST_I32_F32 (opcode 12)

Convert from a single-precision float input to a signed 32-bit integer using round-to-nearest-integer semantics (ignore the default rounding mode) and store the result into a vector register.

### Notes

0.5ULP accuracy, denormals are supported.
