---
mnemonic: V_CMPX_EQ_I32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "194"
pages: [349]
source: RDNA3 ISA PDF p.349-349
---

# V_CMPX_EQ_I32 (opcode 194)

Return 1 iff A equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
