---
mnemonic: V_CMP_NGE_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "41"
pages: [464]
source: RDNA3 ISA PDF p.464-464
---

# V_CMP_NGE_F64 (opcode 41)

Return 1 iff A not greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
