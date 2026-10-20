---
mnemonic: V_CMP_F_U32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "72"
pages: [324]
source: RDNA3 ISA PDF p.324-324
---

# V_CMP_F_U32 (opcode 72)

Return 0.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
