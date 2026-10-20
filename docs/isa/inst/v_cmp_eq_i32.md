---
mnemonic: V_CMP_EQ_I32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "66"
pages: [322]
source: RDNA3 ISA PDF p.322-322
---

# V_CMP_EQ_I32 (opcode 66)

Return 1 iff A equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
