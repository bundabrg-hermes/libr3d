---
mnemonic: V_CMPX_LG_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "149"
pages: [486]
source: RDNA3 ISA PDF p.486-486
---

# V_CMPX_LG_F32 (opcode 149)

Return 1 iff A less than or greater than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
