---
mnemonic: V_CMP_NLT_F64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "46"
pages: [318]
source: RDNA3 ISA PDF p.318-318
---

# V_CMP_NLT_F64 (opcode 46)

Return 1 iff A not less than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
