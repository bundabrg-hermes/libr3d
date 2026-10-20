---
mnemonic: V_CMPX_GE_U32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "206"
pages: [352]
source: RDNA3 ISA PDF p.352-352
---

# V_CMPX_GE_U32 (opcode 206)

Return 1 iff A greater than or equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
