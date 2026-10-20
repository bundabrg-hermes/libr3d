---
mnemonic: V_CVT_PK_RTZ_F16_F32
section: "16.7. VOP2 Instructions"
format: VOP2
opcode: "47"
pages: [269]
source: RDNA3 ISA PDF p.269-269
---

# V_CVT_PK_RTZ_F16_F32 (opcode 47)

Convert two single-precision float inputs into a packed FP16 result with round toward zero semantics (ignore the current rounding mode), and store the result into a vector register.

### Notes

This opcode is intended for use with 16-bit compressed exports. See V_CVT_F16_F32 for a version that respects the current rounding mode.
