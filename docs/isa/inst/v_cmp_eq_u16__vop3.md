---
mnemonic: V_CMP_EQ_U16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "58"
pages: [468]
source: RDNA3 ISA PDF p.468-468
---

# V_CMP_EQ_U16 (opcode 58)

Return 1 iff A equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
