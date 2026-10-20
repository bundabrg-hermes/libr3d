---
mnemonic: V_CMPX_O_F64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "167"
pages: [343]
source: RDNA3 ISA PDF p.343-343
---

# V_CMPX_O_F64 (opcode 167)

Return 1 iff A orderable with B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
