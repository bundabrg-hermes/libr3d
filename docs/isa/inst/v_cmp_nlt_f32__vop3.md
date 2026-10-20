---
mnemonic: V_CMP_NLT_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "30"
pages: [461]
source: RDNA3 ISA PDF p.461-461
---

# V_CMP_NLT_F32 (opcode 30)

Return 1 iff A not less than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
