---
mnemonic: V_CMP_NE_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "77"
pages: [473]
source: RDNA3 ISA PDF p.473-473
---

# V_CMP_NE_U32 (opcode 77)

Return 1 iff A not equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
