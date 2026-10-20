---
mnemonic: V_CMP_NGT_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "11"
pages: [456]
source: RDNA3 ISA PDF p.456-456
---

# V_CMP_NGT_F16 (opcode 11)

Return 1 iff A not greater than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
