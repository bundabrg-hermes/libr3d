---
mnemonic: V_CMP_LT_I64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "81"
pages: [326]
source: RDNA3 ISA PDF p.326-326
---

# V_CMP_LT_I64 (opcode 81)

Return 1 iff A less than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
