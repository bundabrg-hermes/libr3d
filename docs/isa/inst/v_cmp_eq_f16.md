---
mnemonic: V_CMP_EQ_F16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "2"
pages: [306]
source: RDNA3 ISA PDF p.306-306
---

# V_CMP_EQ_F16 (opcode 2)

Return 1 iff A equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
