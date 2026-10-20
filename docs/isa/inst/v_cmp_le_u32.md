---
mnemonic: V_CMP_LE_U32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "75"
pages: [325]
source: RDNA3 ISA PDF p.325-325
---

# V_CMP_LE_U32 (opcode 75)

Return 1 iff A less than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
