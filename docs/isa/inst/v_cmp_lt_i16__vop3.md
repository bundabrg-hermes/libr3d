---
mnemonic: V_CMP_LT_I16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "49"
pages: [466]
source: RDNA3 ISA PDF p.466-466
---

# V_CMP_LT_I16 (opcode 49)

Return 1 iff A less than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
