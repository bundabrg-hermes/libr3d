---
mnemonic: V_CMPX_U_F64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "168"
pages: [343]
source: RDNA3 ISA PDF p.343-343
---

# V_CMPX_U_F64 (opcode 168)

Return 1 iff A not orderable with B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
