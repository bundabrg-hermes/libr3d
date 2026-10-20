---
mnemonic: V_CMP_LE_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "19"
pages: [310, 311]
source: RDNA3 ISA PDF p.310-311
---

# V_CMP_LE_F32 (opcode 19)

Return 1 iff A less than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
