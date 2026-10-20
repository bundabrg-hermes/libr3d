---
mnemonic: V_MUL_I32_I24
section: "16.7. VOP2 Instructions"
format: VOP2
opcode: "9"
pages: [261]
source: RDNA3 ISA PDF p.261-261
---

# V_MUL_I32_I24 (opcode 9)

Multiply two signed 24 bit integer inputs and store the result as a signed 32 bit integer into a vector register.

### Notes

This opcode is expected to be as efficient as basic single-precision opcodes since it utilizes the single-precision floating point multiplier. See also V_MUL_HI_I32_I24.
