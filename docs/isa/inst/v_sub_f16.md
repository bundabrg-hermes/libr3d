---
mnemonic: V_SUB_F16
section: "16.7. VOP2 Instructions"
format: VOP2
opcode: "51"
pages: [269]
source: RDNA3 ISA PDF p.269-269
---

# V_SUB_F16 (opcode 51)

Subtract the second floating point input from the first input and store the result into a vector register.

### Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.
