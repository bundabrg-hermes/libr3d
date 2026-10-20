---
mnemonic: V_CMP_GE_I16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "54"
pages: [320]
source: RDNA3 ISA PDF p.320-320
---

# V_CMP_GE_I16 (opcode 54)

Return 1 iff A greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
