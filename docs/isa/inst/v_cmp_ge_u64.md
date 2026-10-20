---
mnemonic: V_CMP_GE_U64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "94"
pages: [330]
source: RDNA3 ISA PDF p.330-330
---

# V_CMP_GE_U64 (opcode 94)

Return 1 iff A greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
