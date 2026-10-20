---
mnemonic: V_CMPX_EQ_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "146"
pages: [485]
source: RDNA3 ISA PDF p.485-485
---

# V_CMPX_EQ_F32 (opcode 146)

Return 1 iff A equal to B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
