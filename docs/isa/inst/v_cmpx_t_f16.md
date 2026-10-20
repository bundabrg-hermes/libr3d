---
mnemonic: V_CMPX_T_F16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "143"
pages: [337]
source: RDNA3 ISA PDF p.337-337
---

# V_CMPX_T_F16 (opcode 143)

Return 1.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
