---
mnemonic: V_CMPX_F_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "144"
pages: [337]
source: RDNA3 ISA PDF p.337-337
---

# V_CMPX_F_F32 (opcode 144)

Return 0.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
