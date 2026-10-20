---
mnemonic: V_CMPX_NGT_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "139"
pages: [483]
source: RDNA3 ISA PDF p.483-483
---

# V_CMPX_NGT_F16 (opcode 139)

Return 1 iff A not greater than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
