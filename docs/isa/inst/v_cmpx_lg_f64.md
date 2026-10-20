---
mnemonic: V_CMPX_LG_F64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "165"
pages: [342]
source: RDNA3 ISA PDF p.342-342
---

# V_CMPX_LG_F64 (opcode 165)

Return 1 iff A less than or greater than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
