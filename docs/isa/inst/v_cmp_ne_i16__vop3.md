---
mnemonic: V_CMP_NE_I16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "53"
pages: [467]
source: RDNA3 ISA PDF p.467-467
---

# V_CMP_NE_I16 (opcode 53)

Return 1 iff A not equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
