---
mnemonic: V_CMP_EQ_I32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "66"
pages: [470]
source: RDNA3 ISA PDF p.470-470
---

# V_CMP_EQ_I32 (opcode 66)

Return 1 iff A equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
