---
mnemonic: V_CMPX_NLE_F16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "140"
pages: [336]
source: RDNA3 ISA PDF p.336-336
---

# V_CMPX_NLE_F16 (opcode 140)

Return 1 iff A not less than or equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
