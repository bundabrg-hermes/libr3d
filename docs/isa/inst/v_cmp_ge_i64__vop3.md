---
mnemonic: V_CMP_GE_I64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "86"
pages: [475]
source: RDNA3 ISA PDF p.475-475
---

# V_CMP_GE_I64 (opcode 86)

Return 1 iff A greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
