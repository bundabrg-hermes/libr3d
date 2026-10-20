---
mnemonic: V_CMP_EQ_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "18"
pages: [458]
source: RDNA3 ISA PDF p.458-458
---

# V_CMP_EQ_F32 (opcode 18)

Return 1 iff A equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
