---
mnemonic: V_CMP_LT_I16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "49"
pages: [318, 319]
source: RDNA3 ISA PDF p.318-319
---

# V_CMP_LT_I16 (opcode 49)

Return 1 iff A less than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
