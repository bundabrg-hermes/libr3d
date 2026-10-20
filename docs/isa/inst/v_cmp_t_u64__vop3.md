---
mnemonic: V_CMP_T_U64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "95"
pages: [478]
source: RDNA3 ISA PDF p.478-478
---

# V_CMP_T_U64 (opcode 95)

Return 1.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
