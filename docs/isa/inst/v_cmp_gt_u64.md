---
mnemonic: V_CMP_GT_U64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "92"
pages: [329]
source: RDNA3 ISA PDF p.329-329
---

# V_CMP_GT_U64 (opcode 92)

Return 1 iff A greater than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
