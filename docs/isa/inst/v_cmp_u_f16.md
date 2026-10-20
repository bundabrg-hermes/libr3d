---
mnemonic: V_CMP_U_F16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "8"
pages: [307, 308]
source: RDNA3 ISA PDF p.307-308
---

# V_CMP_U_F16 (opcode 8)

Return 1 iff A not orderable with B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
