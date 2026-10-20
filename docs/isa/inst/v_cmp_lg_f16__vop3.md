---
mnemonic: V_CMP_LG_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "5"
pages: [454]
source: RDNA3 ISA PDF p.454-454
---

# V_CMP_LG_F16 (opcode 5)

Return 1 iff A less than or greater than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
