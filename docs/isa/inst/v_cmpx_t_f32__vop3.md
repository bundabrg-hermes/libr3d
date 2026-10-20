---
mnemonic: V_CMPX_T_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "159"
pages: [489]
source: RDNA3 ISA PDF p.489-489
---

# V_CMPX_T_F32 (opcode 159)

Return 1.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
