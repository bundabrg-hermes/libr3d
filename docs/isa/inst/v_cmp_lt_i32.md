---
mnemonic: V_CMP_LT_I32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "65"
pages: [322]
source: RDNA3 ISA PDF p.322-322
---

# V_CMP_LT_I32 (opcode 65)

Return 1 iff A less than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
