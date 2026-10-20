---
mnemonic: V_CMP_NE_U64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "93"
pages: [477]
source: RDNA3 ISA PDF p.477-477
---

# V_CMP_NE_U64 (opcode 93)

Return 1 iff A not equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
