---
mnemonic: V_CMP_LE_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "35"
pages: [462, 463]
source: RDNA3 ISA PDF p.462-463
---

# V_CMP_LE_F64 (opcode 35)

Return 1 iff A less than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
