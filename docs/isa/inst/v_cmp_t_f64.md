---
mnemonic: V_CMP_T_F64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "47"
pages: [318]
source: RDNA3 ISA PDF p.318-318
---

# V_CMP_T_F64 (opcode 47)

Return 1.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
