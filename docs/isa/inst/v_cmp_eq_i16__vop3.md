---
mnemonic: V_CMP_EQ_I16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "50"
pages: [466]
source: RDNA3 ISA PDF p.466-466
---

# V_CMP_EQ_I16 (opcode 50)

Return 1 iff A equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
