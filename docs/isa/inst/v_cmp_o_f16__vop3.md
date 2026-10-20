---
mnemonic: V_CMP_O_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "7"
pages: [455]
source: RDNA3 ISA PDF p.455-455
---

# V_CMP_O_F16 (opcode 7)

Return 1 iff A orderable with B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
