---
mnemonic: V_CMPX_T_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "159"
pages: [341]
source: RDNA3 ISA PDF p.341-341
---

# V_CMPX_T_F32 (opcode 159)

Return 1.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
