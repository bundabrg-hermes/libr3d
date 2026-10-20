---
mnemonic: V_CMP_GT_I16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "52"
pages: [319]
source: RDNA3 ISA PDF p.319-319
---

# V_CMP_GT_I16 (opcode 52)

Return 1 iff A greater than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
