---
mnemonic: V_CMP_GT_I16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "52"
pages: [467]
source: RDNA3 ISA PDF p.467-467
---

# V_CMP_GT_I16 (opcode 52)

Return 1 iff A greater than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
