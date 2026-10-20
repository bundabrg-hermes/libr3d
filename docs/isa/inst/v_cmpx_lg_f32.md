---
mnemonic: V_CMPX_LG_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "149"
pages: [338]
source: RDNA3 ISA PDF p.338-338
---

# V_CMPX_LG_F32 (opcode 149)

Return 1 iff A less than or greater than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
