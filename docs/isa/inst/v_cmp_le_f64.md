---
mnemonic: V_CMP_LE_F64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "35"
pages: [315]
source: RDNA3 ISA PDF p.315-315
---

# V_CMP_LE_F64 (opcode 35)

Return 1 iff A less than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
