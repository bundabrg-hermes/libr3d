---
mnemonic: V_CMP_NE_I16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "53"
pages: [319, 320]
source: RDNA3 ISA PDF p.319-320
---

# V_CMP_NE_I16 (opcode 53)

Return 1 iff A not equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
