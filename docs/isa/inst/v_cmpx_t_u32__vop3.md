---
mnemonic: V_CMPX_T_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "207"
pages: [500]
source: RDNA3 ISA PDF p.500-500
---

# V_CMPX_T_U32 (opcode 207)

Return 1.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
