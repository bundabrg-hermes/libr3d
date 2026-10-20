---
mnemonic: V_CMPX_LT_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "201"
pages: [498]
source: RDNA3 ISA PDF p.498-498
---

# V_CMPX_LT_U32 (opcode 201)

Return 1 iff A less than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
