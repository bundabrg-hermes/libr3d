---
mnemonic: V_CMP_T_I64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "87"
pages: [475, 476]
source: RDNA3 ISA PDF p.475-476
---

# V_CMP_T_I64 (opcode 87)

Return 1.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
