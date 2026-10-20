---
mnemonic: V_CMP_GE_I16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "54"
pages: [467, 468]
source: RDNA3 ISA PDF p.467-468
---

# V_CMP_GE_I16 (opcode 54)

Return 1 iff A greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
