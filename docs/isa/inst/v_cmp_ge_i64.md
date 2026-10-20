---
mnemonic: V_CMP_GE_I64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "86"
pages: [328]
source: RDNA3 ISA PDF p.328-328
---

# V_CMP_GE_I64 (opcode 86)

Return 1 iff A greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
