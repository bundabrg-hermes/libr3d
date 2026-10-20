---
mnemonic: V_CMPX_LE_I16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "179"
pages: [493, 494]
source: RDNA3 ISA PDF p.493-494
---

# V_CMPX_LE_I16 (opcode 179)

Return 1 iff A less than or equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
