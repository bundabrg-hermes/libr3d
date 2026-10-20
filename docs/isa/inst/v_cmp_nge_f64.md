---
mnemonic: V_CMP_NGE_F64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "41"
pages: [316, 317]
source: RDNA3 ISA PDF p.316-317
---

# V_CMP_NGE_F64 (opcode 41)

Return 1 iff A not greater than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
