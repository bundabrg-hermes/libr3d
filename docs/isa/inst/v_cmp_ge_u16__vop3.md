---
mnemonic: V_CMP_GE_U16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "62"
pages: [469]
source: RDNA3 ISA PDF p.469-469
---

# V_CMP_GE_U16 (opcode 62)

Return 1 iff A greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
