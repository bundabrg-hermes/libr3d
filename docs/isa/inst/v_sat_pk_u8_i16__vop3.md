---
mnemonic: V_SAT_PK_U8_I16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "482"
pages: [400]
source: RDNA3 ISA PDF p.400-400
---

# V_SAT_PK_U8_I16 (opcode 482)

Given two 16-bit unsigned integer inputs, saturate each input over an 8-bit unsigned range, pack the resulting values into a 16-bit word and store the result into a vector register.

### Notes

Used for 4x16bit data packed as 4x8bit data.
