---
mnemonic: V_CMP_NGE_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "9"
pages: [455]
source: RDNA3 ISA PDF p.455-455
---

# V_CMP_NGE_F16 (opcode 9)

Return 1 iff A not greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
