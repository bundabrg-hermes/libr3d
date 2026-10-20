---
mnemonic: V_CMP_EQ_I64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "82"
pages: [474]
source: RDNA3 ISA PDF p.474-474
---

# V_CMP_EQ_I64 (opcode 82)

Return 1 iff A equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
