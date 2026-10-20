---
mnemonic: V_CMPX_LE_U32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "203"
pages: [351]
source: RDNA3 ISA PDF p.351-351
---

# V_CMPX_LE_U32 (opcode 203)

Return 1 iff A less than or equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
