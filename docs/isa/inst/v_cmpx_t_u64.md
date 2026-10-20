---
mnemonic: V_CMPX_T_U64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "223"
pages: [356]
source: RDNA3 ISA PDF p.356-356
---

# V_CMPX_T_U64 (opcode 223)

Return 1.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
