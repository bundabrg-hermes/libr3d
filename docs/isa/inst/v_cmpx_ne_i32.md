---
mnemonic: V_CMPX_NE_I32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "197"
pages: [349, 350]
source: RDNA3 ISA PDF p.349-350
---

# V_CMPX_NE_I32 (opcode 197)

Return 1 iff A not equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
