---
mnemonic: V_CMPX_GT_U64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "220"
pages: [503]
source: RDNA3 ISA PDF p.503-503
---

# V_CMPX_GT_U64 (opcode 220)

Return 1 iff A greater than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
