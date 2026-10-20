---
mnemonic: V_CMP_NLE_F16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "12"
pages: [308, 309]
source: RDNA3 ISA PDF p.308-309
---

# V_CMP_NLE_F16 (opcode 12)

Return 1 iff A not less than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
