---
mnemonic: V_CVT_I32_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "387"
pages: [375]
source: RDNA3 ISA PDF p.375-375
---

# V_CVT_I32_F64 (opcode 387)

Convert from a double-precision float input to a signed 32-bit integer and store the result into a vector register.

### Notes

0.5ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0. Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this conversion iff CLAMP == 1.
