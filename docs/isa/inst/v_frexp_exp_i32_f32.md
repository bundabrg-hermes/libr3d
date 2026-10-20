---
mnemonic: V_FREXP_EXP_I32_F32
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "63"
pages: [291]
source: RDNA3 ISA PDF p.291-291
---

# V_FREXP_EXP_I32_F32 (opcode 63)

Extract the exponent of a single-precision float input and store the result as a signed 32-bit integer into a vector register.

### Notes

This operation satisfies the invariant S0.f32 = significand * (2 ** exponent). See also V_FREXP_MANT_F32, which returns the significand. See the C library function frexp() for more information.
