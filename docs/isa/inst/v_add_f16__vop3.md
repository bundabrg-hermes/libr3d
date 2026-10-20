---
mnemonic: V_ADD_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "306"
pages: [411]
source: RDNA3 ISA PDF p.411-411
---

# V_ADD_F16 (opcode 306)

Add two floating point inputs and store the result into a vector register.

### Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.
