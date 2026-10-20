---
mnemonic: V_CVT_NEAREST_I32_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "396"
pages: [377, 378]
source: RDNA3 ISA PDF p.377-378
---

# V_CVT_NEAREST_I32_F32 (opcode 396)

Convert from a single-precision float input to a signed 32-bit integer using round-to-nearest-integer semantics (ignore the default rounding mode) and store the result into a vector register.

### Notes

0.5ULP accuracy, denormals are supported.
