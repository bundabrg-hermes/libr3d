---
mnemonic: V_CMPX_NEQ_F64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "173"
pages: [344, 345]
source: RDNA3 ISA PDF p.344-345
---

# V_CMPX_NEQ_F64 (opcode 173)

Return 1 iff A not equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
