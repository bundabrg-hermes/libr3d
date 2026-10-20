---
mnemonic: V_CMP_EQ_I64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "82"
pages: [326, 327]
source: RDNA3 ISA PDF p.326-327
---

# V_CMP_EQ_I64 (opcode 82)

Return 1 iff A equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
