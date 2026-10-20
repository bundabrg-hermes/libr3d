---
mnemonic: V_CMP_EQ_U64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "90"
pages: [329]
source: RDNA3 ISA PDF p.329-329
---

# V_CMP_EQ_U64 (opcode 90)

Return 1 iff A equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
