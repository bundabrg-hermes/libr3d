---
mnemonic: V_CMPX_T_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "143"
pages: [484, 485]
source: RDNA3 ISA PDF p.484-485
---

# V_CMPX_T_F16 (opcode 143)

Return 1.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
