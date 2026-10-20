---
mnemonic: V_CMPX_GE_U16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "190"
pages: [496]
source: RDNA3 ISA PDF p.496-496
---

# V_CMPX_GE_U16 (opcode 190)

Return 1 iff A greater than or equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
