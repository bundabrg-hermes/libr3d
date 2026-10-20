---
mnemonic: V_CMPX_O_F16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "135"
pages: [335]
source: RDNA3 ISA PDF p.335-335
---

# V_CMPX_O_F16 (opcode 135)

Return 1 iff A orderable with B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
