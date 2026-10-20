---
mnemonic: V_CMP_GT_I32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "68"
pages: [323]
source: RDNA3 ISA PDF p.323-323
---

# V_CMP_GT_I32 (opcode 68)

Return 1 iff A greater than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
