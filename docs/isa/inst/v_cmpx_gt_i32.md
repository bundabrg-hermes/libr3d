---
mnemonic: V_CMPX_GT_I32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "196"
pages: [349]
source: RDNA3 ISA PDF p.349-349
---

# V_CMPX_GT_I32 (opcode 196)

Return 1 iff A greater than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
