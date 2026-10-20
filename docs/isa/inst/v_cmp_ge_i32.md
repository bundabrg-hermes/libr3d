---
mnemonic: V_CMP_GE_I32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "70"
pages: [323]
source: RDNA3 ISA PDF p.323-323
---

# V_CMP_GE_I32 (opcode 70)

Return 1 iff A greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
