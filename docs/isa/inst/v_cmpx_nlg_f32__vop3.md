---
mnemonic: V_CMPX_NLG_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "154"
pages: [487]
source: RDNA3 ISA PDF p.487-487
---

# V_CMPX_NLG_F32 (opcode 154)

Return 1 iff A not less than or greater than B.

### Notes

Write only EXEC. SDST must be set to EXEC_LO. Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
