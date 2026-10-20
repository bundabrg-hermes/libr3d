---
mnemonic: V_CMPX_F_F16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "128"
pages: [333]
source: RDNA3 ISA PDF p.333-333
---

# V_CMPX_F_F16 (opcode 128)

Return 0.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
