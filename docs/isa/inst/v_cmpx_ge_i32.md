---
mnemonic: V_CMPX_GE_I32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "198"
pages: [350]
source: RDNA3 ISA PDF p.350-350
---

# V_CMPX_GE_I32 (opcode 198)

Return 1 iff A greater than or equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
