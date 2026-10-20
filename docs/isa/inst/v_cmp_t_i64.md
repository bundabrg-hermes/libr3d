---
mnemonic: V_CMP_T_I64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "87"
pages: [328]
source: RDNA3 ISA PDF p.328-328
---

# V_CMP_T_I64 (opcode 87)

Return 1.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
