---
mnemonic: V_CMPX_NE_I64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "213"
pages: [353, 354]
source: RDNA3 ISA PDF p.353-354
---

# V_CMPX_NE_I64 (opcode 213)

Return 1 iff A not equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
