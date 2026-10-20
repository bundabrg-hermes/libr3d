---
mnemonic: V_CMPX_F_I64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "208"
pages: [500]
source: RDNA3 ISA PDF p.500-500
---

# V_CMPX_F_I64 (opcode 208)

Return 0.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
