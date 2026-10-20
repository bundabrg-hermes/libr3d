---
mnemonic: V_CMP_T_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "15"
pages: [457]
source: RDNA3 ISA PDF p.457-457
---

# V_CMP_T_F16 (opcode 15)

Return 1.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
