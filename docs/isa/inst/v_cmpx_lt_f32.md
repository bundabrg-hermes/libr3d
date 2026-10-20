---
mnemonic: V_CMPX_LT_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "145"
pages: [337]
source: RDNA3 ISA PDF p.337-337
---

# V_CMPX_LT_F32 (opcode 145)

Return 1 iff A less than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
