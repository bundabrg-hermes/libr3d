---
mnemonic: V_CMPX_NEQ_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "141"
pages: [484]
source: RDNA3 ISA PDF p.484-484
---

# V_CMPX_NEQ_F16 (opcode 141)

Return 1 iff A not equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
