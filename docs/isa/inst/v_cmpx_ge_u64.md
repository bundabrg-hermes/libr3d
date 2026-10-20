---
mnemonic: V_CMPX_GE_U64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "222"
pages: [356]
source: RDNA3 ISA PDF p.356-356
---

# V_CMPX_GE_U64 (opcode 222)

Return 1 iff A greater than or equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
