---
mnemonic: V_MUL_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "309"
pages: [412]
source: RDNA3 ISA PDF p.412-412
---

# V_MUL_F16 (opcode 309)

Multiply two floating point inputs and store the result into a vector register.

### Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.
