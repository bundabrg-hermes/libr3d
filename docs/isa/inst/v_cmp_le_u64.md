---
mnemonic: V_CMP_LE_U64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "91"
pages: [329]
source: RDNA3 ISA PDF p.329-329
---

# V_CMP_LE_U64 (opcode 91)

Return 1 iff A less than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
