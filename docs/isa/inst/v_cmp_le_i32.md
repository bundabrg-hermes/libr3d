---
mnemonic: V_CMP_LE_I32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "67"
pages: [322, 323]
source: RDNA3 ISA PDF p.322-323
---

# V_CMP_LE_I32 (opcode 67)

Return 1 iff A less than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
