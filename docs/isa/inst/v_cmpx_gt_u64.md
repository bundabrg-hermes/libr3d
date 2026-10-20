---
mnemonic: V_CMPX_GT_U64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "220"
pages: [355]
source: RDNA3 ISA PDF p.355-355
---

# V_CMPX_GT_U64 (opcode 220)

Return 1 iff A greater than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
