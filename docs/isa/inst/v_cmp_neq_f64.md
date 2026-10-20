---
mnemonic: V_CMP_NEQ_F64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "45"
pages: [318]
source: RDNA3 ISA PDF p.318-318
---

# V_CMP_NEQ_F64 (opcode 45)

Return 1 iff A not equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
