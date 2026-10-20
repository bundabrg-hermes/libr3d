---
mnemonic: V_CMPX_EQ_U32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "202"
pages: [351]
source: RDNA3 ISA PDF p.351-351
---

# V_CMPX_EQ_U32 (opcode 202)

Return 1 iff A equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
