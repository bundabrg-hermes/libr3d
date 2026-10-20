---
mnemonic: V_CVT_PK_RTZ_F16_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "303"
pages: [411]
source: RDNA3 ISA PDF p.411-411
---

# V_CVT_PK_RTZ_F16_F32 (opcode 303)

Convert two single-precision float inputs into a packed FP16 result with round toward zero semantics (ignore the current rounding mode), and store the result into a vector register.

### Notes

This opcode is intended for use with 16-bit compressed exports. See V_CVT_F16_F32 for a version that respects the current rounding mode.
