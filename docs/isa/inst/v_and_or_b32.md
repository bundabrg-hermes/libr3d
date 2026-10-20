---
mnemonic: V_AND_OR_B32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "599"
pages: [431]
source: RDNA3 ISA PDF p.431-431
---

# V_AND_OR_B32 (opcode 599)

Calculate bitwise AND on the first two vector inputs, then compute the bitwise OR of the intermediate result and the third vector input, then store the result into a vector register.

### Notes

Input and output modifiers not supported.
