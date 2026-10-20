---
mnemonic: V_CMPX_EQ_I64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "210"
pages: [500, 501]
source: RDNA3 ISA PDF p.500-501
---

# V_CMPX_EQ_I64 (opcode 210)

Return 1 iff A equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
