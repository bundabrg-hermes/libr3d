---
mnemonic: V_ADD_LSHL_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "583"
pages: [427]
source: RDNA3 ISA PDF p.427-427
---

# V_ADD_LSHL_U32 (opcode 583)

Add the first two integer inputs, then given a shift count in the third input, calculate the logical shift left of the intermediate result, then store the final result into a vector register.
