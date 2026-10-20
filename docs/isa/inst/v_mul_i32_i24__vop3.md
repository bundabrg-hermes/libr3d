---
mnemonic: V_MUL_I32_I24
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "265"
pages: [403]
source: RDNA3 ISA PDF p.403-403
---

# V_MUL_I32_I24 (opcode 265)

Multiply two signed 24 bit integer inputs and store the result as a signed 32 bit integer into a vector register.

### Notes

This opcode is expected to be as efficient as basic single-precision opcodes since it utilizes the single-precision floating point multiplier. See also V_MUL_HI_I32_I24.
