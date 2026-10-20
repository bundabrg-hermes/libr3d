---
mnemonic: V_CMPX_GT_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "148"
pages: [338]
source: RDNA3 ISA PDF p.338-338
---

# V_CMPX_GT_F32 (opcode 148)

Return 1 iff A greater than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
