---
mnemonic: V_CMP_LG_F16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "5"
pages: [307]
source: RDNA3 ISA PDF p.307-307
---

# V_CMP_LG_F16 (opcode 5)

Return 1 iff A less than or greater than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
