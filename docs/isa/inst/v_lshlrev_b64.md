---
mnemonic: V_LSHLREV_B64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "828"
pages: [450, 451]
source: RDNA3 ISA PDF p.450-451
---

# V_LSHLREV_B64 (opcode 828)

Given a shift count in the first vector input, calculate the logical shift left of the second vector input and store the result into a vector register.

### Notes

Only one scalar broadcast constant is allowed.
