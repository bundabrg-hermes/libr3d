---
mnemonic: V_CMPX_F_I32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "192"
pages: [348]
source: RDNA3 ISA PDF p.348-348
---

# V_CMPX_F_I32 (opcode 192)

Return 0.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
