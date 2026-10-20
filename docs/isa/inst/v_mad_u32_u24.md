---
mnemonic: V_MAD_U32_U24
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "523"
pages: [415]
source: RDNA3 ISA PDF p.415-415
---

# V_MAD_U32_U24 (opcode 523)

Multiply two unsigned 24-bit integers, add an unsigned 32-bit integer and store the result as an unsigned 32-bit integer.

### Notes

This opcode is expected to be as efficient as basic single-precision opcodes since it utilizes the single-precision floating point multiplier.
