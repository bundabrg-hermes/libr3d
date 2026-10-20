---
mnemonic: V_CMP_GT_U32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "76"
pages: [325]
source: RDNA3 ISA PDF p.325-325
---

# V_CMP_GT_U32 (opcode 76)

Return 1 iff A greater than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
