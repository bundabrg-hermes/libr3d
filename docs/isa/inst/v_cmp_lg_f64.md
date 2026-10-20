---
mnemonic: V_CMP_LG_F64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "37"
pages: [315]
source: RDNA3 ISA PDF p.315-315
---

# V_CMP_LG_F64 (opcode 37)

Return 1 iff A less than or greater than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
