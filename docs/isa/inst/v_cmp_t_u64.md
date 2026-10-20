---
mnemonic: V_CMP_T_U64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "95"
pages: [330]
source: RDNA3 ISA PDF p.330-330
---

# V_CMP_T_U64 (opcode 95)

Return 1.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
