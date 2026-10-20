---
mnemonic: V_CMP_NLG_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "42"
pages: [464, 465]
source: RDNA3 ISA PDF p.464-465
---

# V_CMP_NLG_F64 (opcode 42)

Return 1 iff A not less than or greater than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
