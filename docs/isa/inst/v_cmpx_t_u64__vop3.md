---
mnemonic: V_CMPX_T_U64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "223"
pages: [504]
source: RDNA3 ISA PDF p.504-504
---

# V_CMPX_T_U64 (opcode 223)

Return 1.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
