---
mnemonic: V_CMPX_NLT_F16
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "142"
pages: [336, 337]
source: RDNA3 ISA PDF p.336-337
---

# V_CMPX_NLT_F16 (opcode 142)

Return 1 iff A not less than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
