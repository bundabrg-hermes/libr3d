---
mnemonic: V_CMP_NE_I32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "69"
pages: [323]
source: RDNA3 ISA PDF p.323-323
---

# V_CMP_NE_I32 (opcode 69)

Return 1 iff A not equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
