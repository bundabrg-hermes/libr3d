---
mnemonic: V_CMP_GE_I32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "70"
pages: [471]
source: RDNA3 ISA PDF p.471-471
---

# V_CMP_GE_I32 (opcode 70)

Return 1 iff A greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
