---
mnemonic: V_CMPX_EQ_I32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "194"
pages: [496, 497]
source: RDNA3 ISA PDF p.496-497
---

# V_CMPX_EQ_I32 (opcode 194)

Return 1 iff A equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
