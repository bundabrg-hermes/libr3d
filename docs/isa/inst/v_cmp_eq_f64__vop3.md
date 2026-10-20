---
mnemonic: V_CMP_EQ_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "34"
pages: [462]
source: RDNA3 ISA PDF p.462-462
---

# V_CMP_EQ_F64 (opcode 34)

Return 1 iff A equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
