---
mnemonic: V_CMPX_O_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "167"
pages: [491]
source: RDNA3 ISA PDF p.491-491
---

# V_CMPX_O_F64 (opcode 167)

Return 1 iff A orderable with B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
