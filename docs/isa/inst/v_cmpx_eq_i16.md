---
mnemonic: V_CMPX_EQ_I16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "178"
pages: [345, 346]
source: RDNA3 ISA PDF p.345-346
---

# V_CMPX_EQ_I16 (opcode 178)

Return 1 iff A equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
