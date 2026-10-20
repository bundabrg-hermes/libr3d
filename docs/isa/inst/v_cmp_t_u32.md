---
mnemonic: V_CMP_T_U32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "79"
pages: [326]
source: RDNA3 ISA PDF p.326-326
---

# V_CMP_T_U32 (opcode 79)

Return 1.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
