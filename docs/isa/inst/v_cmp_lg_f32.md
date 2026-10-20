---
mnemonic: V_CMP_LG_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "21"
pages: [311]
source: RDNA3 ISA PDF p.311-311
---

# V_CMP_LG_F32 (opcode 21)

Return 1 iff A less than or greater than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
