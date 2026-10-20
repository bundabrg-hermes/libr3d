---
mnemonic: V_CMPX_F_F64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "160"
pages: [341]
source: RDNA3 ISA PDF p.341-341
---

# V_CMPX_F_F64 (opcode 160)

Return 0.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
