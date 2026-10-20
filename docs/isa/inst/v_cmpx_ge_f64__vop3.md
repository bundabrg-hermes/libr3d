---
mnemonic: V_CMPX_GE_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "166"
pages: [490]
source: RDNA3 ISA PDF p.490-490
---

# V_CMPX_GE_F64 (opcode 166)

Return 1 iff A greater than or equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
