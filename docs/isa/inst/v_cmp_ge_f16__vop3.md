---
mnemonic: V_CMP_GE_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "6"
pages: [454, 455]
source: RDNA3 ISA PDF p.454-455
---

# V_CMP_GE_F16 (opcode 6)

Return 1 iff A greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
