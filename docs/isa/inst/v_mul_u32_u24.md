---
mnemonic: V_MUL_U32_U24
section: "16.7. VOP2 Instructions"
format: VOP2
opcode: "11"
pages: [261, 262]
source: RDNA3 ISA PDF p.261-262
---

# V_MUL_U32_U24 (opcode 11)

Multiply two unsigned 24 bit integer inputs and store the result as a unsigned 32 bit integer into a vector register.

### Notes

This opcode is expected to be as efficient as basic single-precision opcodes since it utilizes the single-precision floating point multiplier. See also V_MUL_HI_U32_U24.
