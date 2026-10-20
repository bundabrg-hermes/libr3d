---
mnemonic: V_CLS_I32
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "59"
pages: [289]
source: RDNA3 ISA PDF p.289-289
---

# V_CLS_I32 (opcode 59)

Count the number of leading bits that are the same as the sign bit of a vector input and store the result into a vector register. Store -1 if all input bits are the same.

### Notes

Compare with S_CLS_I32, which performs the equivalent operation in the scalar ALU. Functional examples:
