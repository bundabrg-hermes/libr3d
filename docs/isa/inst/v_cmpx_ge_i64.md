---
mnemonic: V_CMPX_GE_I64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "214"
pages: [354]
source: RDNA3 ISA PDF p.354-354
---

# V_CMPX_GE_I64 (opcode 214)

Return 1 iff A greater than or equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
