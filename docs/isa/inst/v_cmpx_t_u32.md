---
mnemonic: V_CMPX_T_U32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "207"
pages: [352]
source: RDNA3 ISA PDF p.352-352
---

# V_CMPX_T_U32 (opcode 207)

Return 1.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
