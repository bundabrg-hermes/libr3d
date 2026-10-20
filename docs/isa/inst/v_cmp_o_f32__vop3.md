---
mnemonic: V_CMP_O_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "23"
pages: [459]
source: RDNA3 ISA PDF p.459-459
---

# V_CMP_O_F32 (opcode 23)

Return 1 iff A orderable with B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
