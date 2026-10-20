---
mnemonic: V_CMPX_F_I64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "208"
pages: [352]
source: RDNA3 ISA PDF p.352-352
---

# V_CMPX_F_I64 (opcode 208)

Return 0.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
