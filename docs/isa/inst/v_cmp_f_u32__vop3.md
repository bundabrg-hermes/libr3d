---
mnemonic: V_CMP_F_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "72"
pages: [471, 472]
source: RDNA3 ISA PDF p.471-472
---

# V_CMP_F_U32 (opcode 72)

Return 0.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
