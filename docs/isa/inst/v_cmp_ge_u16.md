---
mnemonic: V_CMP_GE_U16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "62"
pages: [321]
source: RDNA3 ISA PDF p.321-321
---

# V_CMP_GE_U16 (opcode 62)

Return 1 iff A greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
