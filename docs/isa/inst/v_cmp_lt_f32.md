---
mnemonic: V_CMP_LT_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "17"
pages: [310]
source: RDNA3 ISA PDF p.310-310
---

# V_CMP_LT_F32 (opcode 17)

Return 1 iff A less than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
