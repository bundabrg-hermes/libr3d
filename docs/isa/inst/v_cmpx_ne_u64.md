---
mnemonic: V_CMPX_NE_U64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "221"
pages: [355, 356]
source: RDNA3 ISA PDF p.355-356
---

# V_CMPX_NE_U64 (opcode 221)

Return 1 iff A not equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
