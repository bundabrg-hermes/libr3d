---
mnemonic: V_SAT_PK_U8_I16
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "98"
pages: [299, 300]
source: RDNA3 ISA PDF p.299-300
---

# V_SAT_PK_U8_I16 (opcode 98)

Given two 16-bit unsigned integer inputs, saturate each input over an 8-bit unsigned range, pack the resulting values into a 16-bit word and store the result into a vector register.

### Notes

Used for 4x16bit data packed as 4x8bit data.
