---
mnemonic: V_CMPX_T_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "175"
pages: [493]
source: RDNA3 ISA PDF p.493-493
---

# V_CMPX_T_F64 (opcode 175)

Return 1.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
