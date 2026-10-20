---
mnemonic: V_CMPX_LE_U16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "187"
pages: [495]
source: RDNA3 ISA PDF p.495-495
---

# V_CMPX_LE_U16 (opcode 187)

Return 1 iff A less than or equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
