---
mnemonic: V_CMPX_NE_I16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "181"
pages: [494]
source: RDNA3 ISA PDF p.494-494
---

# V_CMPX_NE_I16 (opcode 181)

Return 1 iff A not equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
