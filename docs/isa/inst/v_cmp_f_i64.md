---
mnemonic: V_CMP_F_I64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "80"
pages: [326]
source: RDNA3 ISA PDF p.326-326
---

# V_CMP_F_I64 (opcode 80)

Return 0.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
