---
mnemonic: V_CMP_F_I64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "80"
pages: [474]
source: RDNA3 ISA PDF p.474-474
---

# V_CMP_F_I64 (opcode 80)

Return 0.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
