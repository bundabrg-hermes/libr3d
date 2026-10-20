---
mnemonic: V_CMP_GE_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "38"
pages: [463]
source: RDNA3 ISA PDF p.463-463
---

# V_CMP_GE_F64 (opcode 38)

Return 1 iff A greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
