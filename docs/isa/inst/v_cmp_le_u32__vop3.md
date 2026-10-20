---
mnemonic: V_CMP_LE_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "75"
pages: [472]
source: RDNA3 ISA PDF p.472-472
---

# V_CMP_LE_U32 (opcode 75)

Return 1 iff A less than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
