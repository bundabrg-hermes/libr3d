---
mnemonic: V_CMPX_NLG_F32
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "154"
pages: [339, 340]
source: RDNA3 ISA PDF p.339-340
---

# V_CMPX_NLG_F32 (opcode 154)

Return 1 iff A not less than or greater than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
