---
mnemonic: V_CMP_GE_F16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "6"
pages: [307]
source: RDNA3 ISA PDF p.307-307
---

# V_CMP_GE_F16 (opcode 6)

Return 1 iff A greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
