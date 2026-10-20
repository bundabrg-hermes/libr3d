---
mnemonic: V_CMPX_LT_I64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "209"
pages: [352, 353]
source: RDNA3 ISA PDF p.352-353
---

# V_CMPX_LT_I64 (opcode 209)

Return 1 iff A less than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
