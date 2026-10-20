---
mnemonic: V_CMP_EQ_U64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "90"
pages: [476]
source: RDNA3 ISA PDF p.476-476
---

# V_CMP_EQ_U64 (opcode 90)

Return 1 iff A equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
