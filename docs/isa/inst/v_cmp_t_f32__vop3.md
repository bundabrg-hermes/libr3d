---
mnemonic: V_CMP_T_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "31"
pages: [461, 462]
source: RDNA3 ISA PDF p.461-462
---

# V_CMP_T_F32 (opcode 31)

Return 1.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
