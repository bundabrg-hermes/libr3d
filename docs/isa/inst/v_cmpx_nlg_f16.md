---
mnemonic: V_CMPX_NLG_F16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "138"
pages: [335]
source: RDNA3 ISA PDF p.335-335
---

# V_CMPX_NLG_F16 (opcode 138)

Return 1 iff A not less than or greater than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
