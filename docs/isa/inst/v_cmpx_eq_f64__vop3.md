---
mnemonic: V_CMPX_EQ_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "162"
pages: [489]
source: RDNA3 ISA PDF p.489-489
---

# V_CMPX_EQ_F64 (opcode 162)

Return 1 iff A equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
