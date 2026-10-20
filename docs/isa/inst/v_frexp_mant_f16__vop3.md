---
mnemonic: V_FREXP_MANT_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "473"
pages: [397]
source: RDNA3 ISA PDF p.397-397
---

# V_FREXP_MANT_F16 (opcode 473)

Extract the binary significand, or mantissa, of an FP16 float input and store the result as an FP16 float into a vector register.

### Notes

This operation satisfies the invariant S0.f16 = significand * (2 ** exponent). Result range is in (-1.0,-0.5][0.5,1.0) in normal cases. See also V_FREXP_EXP_I_F16, which returns integer exponent. See the C library function frexp() for more information.
