---
mnemonic: V_CMP_LT_U16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "57"
pages: [320]
source: RDNA3 ISA PDF p.320-320
---

# V_CMP_LT_U16 (opcode 57)

Return 1 iff A less than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
