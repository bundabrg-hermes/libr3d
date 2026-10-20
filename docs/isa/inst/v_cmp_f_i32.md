---
mnemonic: V_CMP_F_I32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "64"
pages: [322]
source: RDNA3 ISA PDF p.322-322
---

# V_CMP_F_I32 (opcode 64)

Return 0.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
