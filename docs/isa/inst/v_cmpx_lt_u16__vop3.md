---
mnemonic: V_CMPX_LT_U16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "185"
pages: [494, 495]
source: RDNA3 ISA PDF p.494-495
---

# V_CMPX_LT_U16 (opcode 185)

Return 1 iff A less than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
