---
mnemonic: V_CMPX_LT_F64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "161"
pages: [341]
source: RDNA3 ISA PDF p.341-341
---

# V_CMPX_LT_F64 (opcode 161)

Return 1 iff A less than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
