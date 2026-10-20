---
mnemonic: V_CMPX_F_U64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "216"
pages: [502]
source: RDNA3 ISA PDF p.502-502
---

# V_CMPX_F_U64 (opcode 216)

Return 0.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
