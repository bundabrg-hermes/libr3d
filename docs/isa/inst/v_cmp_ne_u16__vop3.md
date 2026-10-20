---
mnemonic: V_CMP_NE_U16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "61"
pages: [469]
source: RDNA3 ISA PDF p.469-469
---

# V_CMP_NE_U16 (opcode 61)

Return 1 iff A not equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
