---
mnemonic: V_CMP_LT_F64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "33"
pages: [314]
source: RDNA3 ISA PDF p.314-314
---

# V_CMP_LT_F64 (opcode 33)

Return 1 iff A less than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
