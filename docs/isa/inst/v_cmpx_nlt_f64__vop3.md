---
mnemonic: V_CMPX_NLT_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "174"
pages: [492, 493]
source: RDNA3 ISA PDF p.492-493
---

# V_CMPX_NLT_F64 (opcode 174)

Return 1 iff A not less than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
