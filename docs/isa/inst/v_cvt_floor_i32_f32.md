---
mnemonic: V_CVT_FLOOR_I32_F32
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "13"
pages: [277, 278]
source: RDNA3 ISA PDF p.277-278
---

# V_CVT_FLOOR_I32_F32 (opcode 13)

Convert from a single-precision float input to a signed 32-bit integer using round-down semantics (ignore the default rounding mode) and store the result into a vector register.

### Notes

1ULP accuracy, denormals are supported.
