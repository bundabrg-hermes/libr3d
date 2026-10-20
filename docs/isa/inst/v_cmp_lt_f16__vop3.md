---
mnemonic: V_CMP_LT_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "1"
pages: [453]
source: RDNA3 ISA PDF p.453-453
---

# V_CMP_LT_F16 (opcode 1)

Return 1 iff A less than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
