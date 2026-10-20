---
mnemonic: V_CMP_NLG_F16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "10"
pages: [308]
source: RDNA3 ISA PDF p.308-308
---

# V_CMP_NLG_F16 (opcode 10)

Return 1 iff A not less than or greater than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
