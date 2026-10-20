---
mnemonic: V_CMP_NLG_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "10"
pages: [456]
source: RDNA3 ISA PDF p.456-456
---

# V_CMP_NLG_F16 (opcode 10)

Return 1 iff A not less than or greater than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
