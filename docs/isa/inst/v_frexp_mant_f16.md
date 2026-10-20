---
mnemonic: V_FREXP_MANT_F16
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "89"
pages: [296, 297]
source: RDNA3 ISA PDF p.296-297
---

# V_FREXP_MANT_F16 (opcode 89)

Extract the binary significand, or mantissa, of an FP16 float input and store the result as an FP16 float into a vector register.

### Notes

This operation satisfies the invariant S0.f16 = significand * (2 ** exponent). Result range is in (-1.0,-0.5][0.5,1.0) in normal cases. See also V_FREXP_EXP_I_F16, which returns integer exponent. See the C library function frexp() for more information.
