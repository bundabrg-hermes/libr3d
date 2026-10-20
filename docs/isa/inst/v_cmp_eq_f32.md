---
mnemonic: V_CMP_EQ_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "18"
pages: [310]
source: RDNA3 ISA PDF p.310-310
---

# V_CMP_EQ_F32 (opcode 18)

Return 1 iff A equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
