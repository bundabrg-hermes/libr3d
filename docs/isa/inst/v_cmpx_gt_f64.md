---
mnemonic: V_CMPX_GT_F64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "164"
pages: [342]
source: RDNA3 ISA PDF p.342-342
---

# V_CMPX_GT_F64 (opcode 164)

Return 1 iff A greater than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
