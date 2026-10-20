---
mnemonic: V_CMP_NGT_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "27"
pages: [460]
source: RDNA3 ISA PDF p.460-460
---

# V_CMP_NGT_F32 (opcode 27)

Return 1 iff A not greater than B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
