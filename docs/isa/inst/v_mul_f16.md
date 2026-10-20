---
mnemonic: V_MUL_F16
section: "16.7. VOP2 Instructions"
format: VOP2
opcode: "53"
pages: [270]
source: RDNA3 ISA PDF p.270-270
---

# V_MUL_F16 (opcode 53)

Multiply two floating point inputs and store the result into a vector register.

### Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.
