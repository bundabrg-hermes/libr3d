---
mnemonic: V_CMPX_F_U64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "216"
pages: [354]
source: RDNA3 ISA PDF p.354-354
---

# V_CMPX_F_U64 (opcode 216)

Return 0.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
