---
mnemonic: V_CMPX_GT_U32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "204"
pages: [351]
source: RDNA3 ISA PDF p.351-351
---

# V_CMPX_GT_U32 (opcode 204)

Return 1 iff A greater than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
