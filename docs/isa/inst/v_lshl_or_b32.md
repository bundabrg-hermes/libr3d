---
mnemonic: V_LSHL_OR_B32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "598"
pages: [431]
source: RDNA3 ISA PDF p.431-431
---

# V_LSHL_OR_B32 (opcode 598)

Given a shift count in the second input, calculate the logical shift left of the first input, then calculate the bitwise OR of the intermediate result and the third input, then store the final result into a vector register.
