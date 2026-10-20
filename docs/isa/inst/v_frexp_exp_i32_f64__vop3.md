---
mnemonic: V_FREXP_EXP_I32_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "444"
pages: [390]
source: RDNA3 ISA PDF p.390-390
---

# V_FREXP_EXP_I32_F64 (opcode 444)

Extract the exponent of a double-precision float input and store the result as a signed 32-bit integer into a vector register.

### Notes

This operation satisfies the invariant S0.f64 = significand * (2 ** exponent). See also V_FREXP_MANT_F64, which returns the significand. See the C library function frexp() for more information.
