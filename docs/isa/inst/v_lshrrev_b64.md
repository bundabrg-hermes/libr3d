---
mnemonic: V_LSHRREV_B64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "829"
pages: [451]
source: RDNA3 ISA PDF p.451-451
---

# V_LSHRREV_B64 (opcode 829)

Given a shift count in the first vector input, calculate the logical shift right of the second vector input and store the result into a vector register.

### Notes

Only one scalar broadcast constant is allowed.
