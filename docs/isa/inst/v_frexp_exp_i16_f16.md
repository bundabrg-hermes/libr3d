---
mnemonic: V_FREXP_EXP_I16_F16
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "90"
pages: [297]
source: RDNA3 ISA PDF p.297-297
---

# V_FREXP_EXP_I16_F16 (opcode 90)

Extract the exponent of an FP16 float input and store the result as a signed 16-bit integer into a vector register.

### Notes

This operation satisfies the invariant S0.f16 = significand * (2 ** exponent). See also V_FREXP_MANT_F16, which returns the significand. See the C library function frexp() for more information.
