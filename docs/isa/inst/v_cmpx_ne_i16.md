---
mnemonic: V_CMPX_NE_I16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "181"
pages: [346]
source: RDNA3 ISA PDF p.346-346
---

# V_CMPX_NE_I16 (opcode 181)

Return 1 iff A not equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
