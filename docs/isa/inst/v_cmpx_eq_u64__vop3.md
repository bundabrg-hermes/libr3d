---
mnemonic: V_CMPX_EQ_U64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "218"
pages: [502, 503]
source: RDNA3 ISA PDF p.502-503
---

# V_CMPX_EQ_U64 (opcode 218)

Return 1 iff A equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
