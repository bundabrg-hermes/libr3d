---
mnemonic: V_CVT_I32_F64
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "3"
pages: [275]
source: RDNA3 ISA PDF p.275-275
---

# V_CVT_I32_F64 (opcode 3)

Convert from a double-precision float input to a signed 32-bit integer and store the result into a vector register.

### Notes

0.5ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0. Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this conversion iff CLAMP == 1.
