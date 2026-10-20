---
mnemonic: V_CMP_NLE_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "44"
pages: [465]
source: RDNA3 ISA PDF p.465-465
---

# V_CMP_NLE_F64 (opcode 44)

Return 1 iff A not less than or equal to B.

### Notes

Signal 'invalid' on sNAN's, and also on qNAN's if clamp is set.
