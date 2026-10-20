---
mnemonic: V_CMPX_GT_U16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "188"
pages: [347, 348]
source: RDNA3 ISA PDF p.347-348
---

# V_CMPX_GT_U16 (opcode 188)

Return 1 iff A greater than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
