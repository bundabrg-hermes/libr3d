---
mnemonic: V_CMPX_LT_I32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "193"
pages: [348, 349]
source: RDNA3 ISA PDF p.348-349
---

# V_CMPX_LT_I32 (opcode 193)

Return 1 iff A less than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
