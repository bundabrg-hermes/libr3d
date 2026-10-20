---
mnemonic: V_CMPX_F_U32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "200"
pages: [350]
source: RDNA3 ISA PDF p.350-350
---

# V_CMPX_F_U32 (opcode 200)

Return 0.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
