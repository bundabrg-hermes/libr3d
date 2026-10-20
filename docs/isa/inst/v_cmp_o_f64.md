---
mnemonic: V_CMP_O_F64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "39"
pages: [316]
source: RDNA3 ISA PDF p.316-316
---

# V_CMP_O_F64 (opcode 39)

Return 1 iff A orderable with B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
