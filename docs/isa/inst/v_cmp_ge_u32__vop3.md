---
mnemonic: V_CMP_GE_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "78"
pages: [473]
source: RDNA3 ISA PDF p.473-473
---

# V_CMP_GE_U32 (opcode 78)

Return 1 iff A greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
