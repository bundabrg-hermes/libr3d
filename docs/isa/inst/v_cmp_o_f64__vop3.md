---
mnemonic: V_CMP_O_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "39"
pages: [464]
source: RDNA3 ISA PDF p.464-464
---

# V_CMP_O_F64 (opcode 39)

Return 1 iff A orderable with B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
