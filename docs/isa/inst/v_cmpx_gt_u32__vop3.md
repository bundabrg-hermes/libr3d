---
mnemonic: V_CMPX_GT_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "204"
pages: [499]
source: RDNA3 ISA PDF p.499-499
---

# V_CMPX_GT_U32 (opcode 204)

Return 1 iff A greater than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
