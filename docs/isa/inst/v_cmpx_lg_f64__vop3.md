---
mnemonic: V_CMPX_LG_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "165"
pages: [490]
source: RDNA3 ISA PDF p.490-490
---

# V_CMPX_LG_F64 (opcode 165)

Return 1 iff A less than or greater than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
