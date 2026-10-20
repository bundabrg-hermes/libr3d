---
mnemonic: V_FREXP_MANT_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "445"
pages: [390, 391]
source: RDNA3 ISA PDF p.390-391
---

# V_FREXP_MANT_F64 (opcode 445)

Extract the binary significand, or mantissa, of a double-precision float input and store the result as a double- precision float into a vector register.

### Notes

This operation satisfies the invariant S0.f64 = significand * (2 ** exponent). Result range is in (-1.0,-0.5][0.5,1.0) in normal cases. See also V_FREXP_EXP_I_F64, which returns integer exponent. See the C library function frexp() for more information.
