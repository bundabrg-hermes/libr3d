---
mnemonic: V_CMP_T_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "31"
pages: [314]
source: RDNA3 ISA PDF p.314-314
---

# V_CMP_T_F32 (opcode 31)

Return 1.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
