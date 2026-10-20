---
mnemonic: V_CMP_GE_U64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "94"
pages: [477, 478]
source: RDNA3 ISA PDF p.477-478
---

# V_CMP_GE_U64 (opcode 94)

Return 1 iff A greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
