---
mnemonic: V_CMP_GE_F64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "38"
pages: [316]
source: RDNA3 ISA PDF p.316-316
---

# V_CMP_GE_F64 (opcode 38)

Return 1 iff A greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
