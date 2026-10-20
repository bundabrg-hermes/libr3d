---
mnemonic: V_CVT_I32_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "392"
pages: [377]
source: RDNA3 ISA PDF p.377-377
---

# V_CVT_I32_F32 (opcode 392)

Convert from a single-precision float input to a signed 32-bit integer and store the result into a vector register.

### Notes

1ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0. Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this conversion iff CLAMP == 1.
