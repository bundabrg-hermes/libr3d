---
mnemonic: V_CMPX_EQ_U64
section: "16.9. VOPC Instructions"
format: VOPC
opcode: "218"
pages: [355]
source: RDNA3 ISA PDF p.355-355
---

# V_CMPX_EQ_U64 (opcode 218)

Return 1 iff A equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
