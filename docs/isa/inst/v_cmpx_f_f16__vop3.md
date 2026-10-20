---
mnemonic: V_CMPX_F_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "128"
pages: [480, 481]
source: RDNA3 ISA PDF p.480-481
---

# V_CMPX_F_F16 (opcode 128)

Return 0.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
