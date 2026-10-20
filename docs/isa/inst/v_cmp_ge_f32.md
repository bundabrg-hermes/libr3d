---
mnemonic: V_CMP_GE_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "22"
pages: [311]
source: RDNA3 ISA PDF p.311-311
---

# V_CMP_GE_F32 (opcode 22)

Return 1 iff A greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
