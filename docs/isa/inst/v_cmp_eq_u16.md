---
mnemonic: V_CMP_EQ_U16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "58"
pages: [320]
source: RDNA3 ISA PDF p.320-320
---

# V_CMP_EQ_U16 (opcode 58)

Return 1 iff A equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
