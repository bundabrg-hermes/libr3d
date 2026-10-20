---
mnemonic: V_FREXP_MANT_F64
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "61"
pages: [290]
source: RDNA3 ISA PDF p.290-290
---

# V_FREXP_MANT_F64 (opcode 61)

Extract the binary significand, or mantissa, of a double-precision float input and store the result as a double- precision float into a vector register.

### Notes

This operation satisfies the invariant S0.f64 = significand * (2 ** exponent). Result range is in (-1.0,-0.5][0.5,1.0) in normal cases. See also V_FREXP_EXP_I_F64, which returns integer exponent. See the C library function frexp() for more information.
