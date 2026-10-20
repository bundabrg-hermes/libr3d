---
mnemonic: V_FREXP_MANT_F32
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "64"
pages: [291]
source: RDNA3 ISA PDF p.291-291
---

# V_FREXP_MANT_F32 (opcode 64)

Extract the binary significand, or mantissa, of a single-precision float input and store the result as a single- precision float into a vector register.

### Notes

This operation satisfies the invariant S0.f32 = significand * (2 ** exponent). Result range is in (-1.0,-0.5][0.5,1.0) in normal cases. See also V_FREXP_EXP_I_F32, which returns integer exponent. See the C library function frexp() for more information.
