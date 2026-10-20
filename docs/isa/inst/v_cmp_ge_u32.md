---
mnemonic: V_CMP_GE_U32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "78"
pages: [325, 326]
source: RDNA3 ISA PDF p.325-326
---

# V_CMP_GE_U32 (opcode 78)

Return 1 iff A greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
