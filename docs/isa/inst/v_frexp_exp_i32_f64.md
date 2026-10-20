---
mnemonic: V_FREXP_EXP_I32_F64
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "60"
pages: [290]
source: RDNA3 ISA PDF p.290-290
---

# V_FREXP_EXP_I32_F64 (opcode 60)

Extract the exponent of a double-precision float input and store the result as a signed 32-bit integer into a vector register.

### Notes

This operation satisfies the invariant S0.f64 = significand * (2 ** exponent). See also V_FREXP_MANT_F64, which returns the significand. See the C library function frexp() for more information.
