---
mnemonic: V_CMP_U_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "24"
pages: [459, 460]
source: RDNA3 ISA PDF p.459-460
---

# V_CMP_U_F32 (opcode 24)

Return 1 iff A not orderable with B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
