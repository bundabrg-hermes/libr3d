---
mnemonic: V_CMP_NEQ_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "29"
pages: [461]
source: RDNA3 ISA PDF p.461-461
---

# V_CMP_NEQ_F32 (opcode 29)

Return 1 iff A not equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
