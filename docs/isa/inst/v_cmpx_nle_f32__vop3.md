---
mnemonic: V_CMPX_NLE_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "156"
pages: [488]
source: RDNA3 ISA PDF p.488-488
---

# V_CMPX_NLE_F32 (opcode 156)

Return 1 iff A not less than or equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
