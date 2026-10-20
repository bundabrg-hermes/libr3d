---
mnemonic: V_CMP_F_U64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "88"
pages: [328]
source: RDNA3 ISA PDF p.328-328
---

# V_CMP_F_U64 (opcode 88)

Return 0.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
