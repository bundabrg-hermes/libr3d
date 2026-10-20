---
mnemonic: V_LSHL_ADD_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "582"
pages: [427]
source: RDNA3 ISA PDF p.427-427
---

# V_LSHL_ADD_U32 (opcode 582)

Given a shift count in the second input, calculate the logical shift left of the first input, then add the third input to the intermediate result, then store the final result into a vector register.
