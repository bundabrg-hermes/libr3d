---
mnemonic: V_CMPX_NEQ_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "157"
pages: [340]
source: RDNA3 ISA PDF p.340-340
---

# V_CMPX_NEQ_F32 (opcode 157)

Return 1 iff A not equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
