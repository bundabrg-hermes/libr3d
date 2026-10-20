---
mnemonic: V_CMP_LT_U32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "73"
pages: [324]
source: RDNA3 ISA PDF p.324-324
---

# V_CMP_LT_U32 (opcode 73)

Return 1 iff A less than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
