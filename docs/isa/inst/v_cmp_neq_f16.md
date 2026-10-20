---
mnemonic: V_CMP_NEQ_F16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "13"
pages: [309]
source: RDNA3 ISA PDF p.309-309
---

# V_CMP_NEQ_F16 (opcode 13)

Return 1 iff A not equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
