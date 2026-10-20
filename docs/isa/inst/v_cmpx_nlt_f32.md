---
mnemonic: V_CMPX_NLT_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "158"
pages: [340, 341]
source: RDNA3 ISA PDF p.340-341
---

# V_CMPX_NLT_F32 (opcode 158)

Return 1 iff A not less than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
