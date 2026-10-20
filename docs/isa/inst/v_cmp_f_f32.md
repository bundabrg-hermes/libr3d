---
mnemonic: V_CMP_F_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "16"
pages: [310]
source: RDNA3 ISA PDF p.310-310
---

# V_CMP_F_F32 (opcode 16)

Return 0.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
