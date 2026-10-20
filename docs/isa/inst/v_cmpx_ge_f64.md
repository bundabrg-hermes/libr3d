---
mnemonic: V_CMPX_GE_F64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "166"
pages: [343]
source: RDNA3 ISA PDF p.343-343
---

# V_CMPX_GE_F64 (opcode 166)

Return 1 iff A greater than or equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
