---
mnemonic: V_CMP_T_I32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "71"
pages: [323, 324]
source: RDNA3 ISA PDF p.323-324
---

# V_CMP_T_I32 (opcode 71)

Return 1.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
