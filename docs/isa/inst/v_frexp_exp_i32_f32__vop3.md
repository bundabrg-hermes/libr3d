---
mnemonic: V_FREXP_EXP_I32_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "447"
pages: [391]
source: RDNA3 ISA PDF p.391-391
---

# V_FREXP_EXP_I32_F32 (opcode 447)

Extract the exponent of a single-precision float input and store the result as a signed 32-bit integer into a vector register.

### Notes

This operation satisfies the invariant S0.f32 = significand * (2 ** exponent). See also V_FREXP_MANT_F32, which returns the significand. See the C library function frexp() for more information.
