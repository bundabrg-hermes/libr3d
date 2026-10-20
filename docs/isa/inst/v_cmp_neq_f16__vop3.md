---
mnemonic: V_CMP_NEQ_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "13"
pages: [456, 457]
source: RDNA3 ISA PDF p.456-457
---

# V_CMP_NEQ_F16 (opcode 13)

Return 1 iff A not equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
