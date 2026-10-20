---
mnemonic: V_CMPX_GT_F16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "132"
pages: [334]
source: RDNA3 ISA PDF p.334-334
---

# V_CMPX_GT_F16 (opcode 132)

Return 1 iff A greater than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
