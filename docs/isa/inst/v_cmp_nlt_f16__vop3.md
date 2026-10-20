---
mnemonic: V_CMP_NLT_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "14"
pages: [457]
source: RDNA3 ISA PDF p.457-457
---

# V_CMP_NLT_F16 (opcode 14)

Return 1 iff A not less than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
