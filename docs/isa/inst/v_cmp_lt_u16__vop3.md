---
mnemonic: V_CMP_LT_U16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "57"
pages: [468]
source: RDNA3 ISA PDF p.468-468
---

# V_CMP_LT_U16 (opcode 57)

Return 1 iff A less than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
