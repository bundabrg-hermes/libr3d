---
mnemonic: V_CMP_U_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "8"
pages: [455]
source: RDNA3 ISA PDF p.455-455
---

# V_CMP_U_F16 (opcode 8)

Return 1 iff A not orderable with B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
