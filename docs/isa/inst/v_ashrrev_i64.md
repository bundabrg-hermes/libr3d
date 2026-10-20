---
mnemonic: V_ASHRREV_I64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "830"
pages: [451]
source: RDNA3 ISA PDF p.451-451
---

# V_ASHRREV_I64 (opcode 830)

Given a shift count in the first vector input, calculate the arithmetic shift right (preserving sign bit) of the second vector input and store the result into a vector register.

### Notes

Only one scalar broadcast constant is allowed.
