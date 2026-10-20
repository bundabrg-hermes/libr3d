---
mnemonic: V_SUBREV_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "308"
pages: [411]
source: RDNA3 ISA PDF p.411-411
---

# V_SUBREV_F16 (opcode 308)

Subtract the first floating point input from the second input and store the result into a vector register.

### Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.
