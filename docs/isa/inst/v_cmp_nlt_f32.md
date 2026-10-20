---
mnemonic: V_CMP_NLT_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "30"
pages: [313, 314]
source: RDNA3 ISA PDF p.313-314
---

# V_CMP_NLT_F32 (opcode 30)

Return 1 iff A not less than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
