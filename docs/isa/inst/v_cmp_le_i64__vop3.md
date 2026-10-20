---
mnemonic: V_CMP_LE_I64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "83"
pages: [474, 475]
source: RDNA3 ISA PDF p.474-475
---

# V_CMP_LE_I64 (opcode 83)

Return 1 iff A less than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
