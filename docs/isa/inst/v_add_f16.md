---
mnemonic: V_ADD_F16
section: "16.7. VOP2 Instructions"
format: VOP2
opcode: "50"
pages: [269]
source: RDNA3 ISA PDF p.269-269
---

# V_ADD_F16 (opcode 50)

Add two floating point inputs and store the result into a vector register.

### Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.
