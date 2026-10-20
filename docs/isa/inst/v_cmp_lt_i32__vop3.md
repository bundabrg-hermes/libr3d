---
mnemonic: V_CMP_LT_I32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "65"
pages: [470]
source: RDNA3 ISA PDF p.470-470
---

# V_CMP_LT_I32 (opcode 65)

Return 1 iff A less than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
