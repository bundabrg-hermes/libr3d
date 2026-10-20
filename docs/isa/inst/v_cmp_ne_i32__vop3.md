---
mnemonic: V_CMP_NE_I32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "69"
pages: [471]
source: RDNA3 ISA PDF p.471-471
---

# V_CMP_NE_I32 (opcode 69)

Return 1 iff A not equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
