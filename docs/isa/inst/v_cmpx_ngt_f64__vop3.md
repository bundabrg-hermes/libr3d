---
mnemonic: V_CMPX_NGT_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "171"
pages: [492]
source: RDNA3 ISA PDF p.492-492
---

# V_CMPX_NGT_F64 (opcode 171)

Return 1 iff A not greater than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
