---
mnemonic: V_CMP_GT_U16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "60"
pages: [468, 469]
source: RDNA3 ISA PDF p.468-469
---

# V_CMP_GT_U16 (opcode 60)

Return 1 iff A greater than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
