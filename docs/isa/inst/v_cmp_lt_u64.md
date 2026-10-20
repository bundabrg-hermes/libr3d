---
mnemonic: V_CMP_LT_U64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "89"
pages: [328]
source: RDNA3 ISA PDF p.328-328
---

# V_CMP_LT_U64 (opcode 89)

Return 1 iff A less than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
