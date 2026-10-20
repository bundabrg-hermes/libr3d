---
mnemonic: V_CMPX_EQ_U16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "186"
pages: [347]
source: RDNA3 ISA PDF p.347-347
---

# V_CMPX_EQ_U16 (opcode 186)

Return 1 iff A equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
