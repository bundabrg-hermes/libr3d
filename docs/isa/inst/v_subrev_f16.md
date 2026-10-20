---
mnemonic: V_SUBREV_F16
section: "16.7. VOP2 Instructions"
format: VOP2
opcode: "52"
pages: [269, 270]
source: RDNA3 ISA PDF p.269-270
---

# V_SUBREV_F16 (opcode 52)

Subtract the first floating point input from the second input and store the result into a vector register.

### Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.
