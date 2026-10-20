---
mnemonic: V_CMPX_LE_I16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "179"
pages: [346]
source: RDNA3 ISA PDF p.346-346
---

# V_CMPX_LE_I16 (opcode 179)

Return 1 iff A less than or equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
