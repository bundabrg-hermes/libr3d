---
mnemonic: V_CMPX_GE_I16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "182"
pages: [346, 347]
source: RDNA3 ISA PDF p.346-347
---

# V_CMPX_GE_I16 (opcode 182)

Return 1 iff A greater than or equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
