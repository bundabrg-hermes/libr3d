---
mnemonic: V_CMP_NGE_F16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "9"
pages: [308]
source: RDNA3 ISA PDF p.308-308
---

# V_CMP_NGE_F16 (opcode 9)

Return 1 iff A not greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
