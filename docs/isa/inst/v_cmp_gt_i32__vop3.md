---
mnemonic: V_CMP_GT_I32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "68"
pages: [470, 471]
source: RDNA3 ISA PDF p.470-471
---

# V_CMP_GT_I32 (opcode 68)

Return 1 iff A greater than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
