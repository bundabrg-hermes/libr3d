---
mnemonic: V_CMPX_GT_I64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "212"
pages: [353]
source: RDNA3 ISA PDF p.353-353
---

# V_CMPX_GT_I64 (opcode 212)

Return 1 iff A greater than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
