---
mnemonic: V_CMP_LG_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "37"
pages: [463]
source: RDNA3 ISA PDF p.463-463
---

# V_CMP_LG_F64 (opcode 37)

Return 1 iff A less than or greater than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
