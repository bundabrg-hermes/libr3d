---
mnemonic: V_CMP_LE_I64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "83"
pages: [327]
source: RDNA3 ISA PDF p.327-327
---

# V_CMP_LE_I64 (opcode 83)

Return 1 iff A less than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
