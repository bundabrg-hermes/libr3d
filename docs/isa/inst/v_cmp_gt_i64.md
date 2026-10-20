---
mnemonic: V_CMP_GT_I64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "84"
pages: [327]
source: RDNA3 ISA PDF p.327-327
---

# V_CMP_GT_I64 (opcode 84)

Return 1 iff A greater than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
