---
mnemonic: V_CMPX_F_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "160"
pages: [489]
source: RDNA3 ISA PDF p.489-489
---

# V_CMPX_F_F64 (opcode 160)

Return 0.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
