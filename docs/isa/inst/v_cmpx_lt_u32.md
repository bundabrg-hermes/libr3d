---
mnemonic: V_CMPX_LT_U32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "201"
pages: [350, 351]
source: RDNA3 ISA PDF p.350-351
---

# V_CMPX_LT_U32 (opcode 201)

Return 1 iff A less than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
