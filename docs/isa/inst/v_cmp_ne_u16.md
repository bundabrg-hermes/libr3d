---
mnemonic: V_CMP_NE_U16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "61"
pages: [321]
source: RDNA3 ISA PDF p.321-321
---

# V_CMP_NE_U16 (opcode 61)

Return 1 iff A not equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
