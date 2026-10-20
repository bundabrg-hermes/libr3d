---
mnemonic: V_CMP_NE_U64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "93"
pages: [329, 330]
source: RDNA3 ISA PDF p.329-330
---

# V_CMP_NE_U64 (opcode 93)

Return 1 iff A not equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
