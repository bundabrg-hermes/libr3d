---
mnemonic: V_CMPX_EQ_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "146"
pages: [337, 338]
source: RDNA3 ISA PDF p.337-338
---

# V_CMPX_EQ_F32 (opcode 146)

Return 1 iff A equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
