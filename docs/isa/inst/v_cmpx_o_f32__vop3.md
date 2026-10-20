---
mnemonic: V_CMPX_O_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "151"
pages: [486, 487]
source: RDNA3 ISA PDF p.486-487
---

# V_CMPX_O_F32 (opcode 151)

Return 1 iff A orderable with B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
