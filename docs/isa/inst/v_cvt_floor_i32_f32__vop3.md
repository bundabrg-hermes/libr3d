---
mnemonic: V_CVT_FLOOR_I32_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "397"
pages: [378]
source: RDNA3 ISA PDF p.378-378
---

# V_CVT_FLOOR_I32_F32 (opcode 397)

Convert from a single-precision float input to a signed 32-bit integer using round-down semantics (ignore the default rounding mode) and store the result into a vector register.

### Notes

1ULP accuracy, denormals are supported.
