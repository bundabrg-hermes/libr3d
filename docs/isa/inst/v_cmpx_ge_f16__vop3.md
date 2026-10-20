---
mnemonic: V_CMPX_GE_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "134"
pages: [482]
source: RDNA3 ISA PDF p.482-482
---

# V_CMPX_GE_F16 (opcode 134)

Return 1 iff A greater than or equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
