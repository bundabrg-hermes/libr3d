---
mnemonic: V_CMP_NGT_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "27"
pages: [313]
source: RDNA3 ISA PDF p.313-313
---

# V_CMP_NGT_F32 (opcode 27)

Return 1 iff A not greater than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
