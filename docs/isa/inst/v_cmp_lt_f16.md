---
mnemonic: V_CMP_LT_F16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "1"
pages: [305, 306]
source: RDNA3 ISA PDF p.305-306
---

# V_CMP_LT_F16 (opcode 1)

Return 1 iff A less than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
