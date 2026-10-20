---
mnemonic: V_CMP_NE_U32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "77"
pages: [325]
source: RDNA3 ISA PDF p.325-325
---

# V_CMP_NE_U32 (opcode 77)

Return 1 iff A not equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
