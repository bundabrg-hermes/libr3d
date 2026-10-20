---
mnemonic: V_CMPX_T_I64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "215"
pages: [502]
source: RDNA3 ISA PDF p.502-502
---

# V_CMPX_T_I64 (opcode 215)

Return 1.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
