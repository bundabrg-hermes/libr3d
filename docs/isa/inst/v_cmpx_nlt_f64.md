---
mnemonic: V_CMPX_NLT_F64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "174"
pages: [345]
source: RDNA3 ISA PDF p.345-345
---

# V_CMPX_NLT_F64 (opcode 174)

Return 1 iff A not less than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
