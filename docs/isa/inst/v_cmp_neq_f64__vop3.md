---
mnemonic: V_CMP_NEQ_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "45"
pages: [465]
source: RDNA3 ISA PDF p.465-465
---

# V_CMP_NEQ_F64 (opcode 45)

Return 1 iff A not equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
