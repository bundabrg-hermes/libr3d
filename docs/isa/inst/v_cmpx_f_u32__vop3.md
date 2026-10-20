---
mnemonic: V_CMPX_F_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "200"
pages: [498]
source: RDNA3 ISA PDF p.498-498
---

# V_CMPX_F_U32 (opcode 200)

Return 0.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
