---
mnemonic: V_CMPX_T_F64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "175"
pages: [345]
source: RDNA3 ISA PDF p.345-345
---

# V_CMPX_T_F64 (opcode 175)

Return 1.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
