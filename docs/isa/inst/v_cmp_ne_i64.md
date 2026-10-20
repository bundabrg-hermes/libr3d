---
mnemonic: V_CMP_NE_I64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "85"
pages: [327]
source: RDNA3 ISA PDF p.327-327
---

# V_CMP_NE_I64 (opcode 85)

Return 1 iff A not equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
