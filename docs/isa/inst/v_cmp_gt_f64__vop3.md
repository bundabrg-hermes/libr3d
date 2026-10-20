---
mnemonic: V_CMP_GT_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "36"
pages: [463]
source: RDNA3 ISA PDF p.463-463
---

# V_CMP_GT_F64 (opcode 36)

Return 1 iff A greater than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
