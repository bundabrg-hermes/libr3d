---
mnemonic: V_CMP_EQ_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "2"
pages: [453, 454]
source: RDNA3 ISA PDF p.453-454
---

# V_CMP_EQ_F16 (opcode 2)

Return 1 iff A equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
