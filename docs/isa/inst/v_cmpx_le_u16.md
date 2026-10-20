---
mnemonic: V_CMPX_LE_U16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "187"
pages: [347]
source: RDNA3 ISA PDF p.347-347
---

# V_CMPX_LE_U16 (opcode 187)

Return 1 iff A less than or equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
