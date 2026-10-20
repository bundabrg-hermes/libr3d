---
mnemonic: V_CMP_LT_I64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "81"
pages: [474]
source: RDNA3 ISA PDF p.474-474
---

# V_CMP_LT_I64 (opcode 81)

Return 1 iff A less than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
