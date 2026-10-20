---
mnemonic: V_CMPX_LT_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "129"
pages: [481]
source: RDNA3 ISA PDF p.481-481
---

# V_CMPX_LT_F16 (opcode 129)

Return 1 iff A less than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
