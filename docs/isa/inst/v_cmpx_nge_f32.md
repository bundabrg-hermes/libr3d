---
mnemonic: V_CMPX_NGE_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "153"
pages: [339]
source: RDNA3 ISA PDF p.339-339
---

# V_CMPX_NGE_F32 (opcode 153)

Return 1 iff A not greater than or equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
