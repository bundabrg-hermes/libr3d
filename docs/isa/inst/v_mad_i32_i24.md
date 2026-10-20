---
mnemonic: V_MAD_I32_I24
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "522"
pages: [414, 415]
source: RDNA3 ISA PDF p.414-415
---

# V_MAD_I32_I24 (opcode 522)

Multiply two signed 24-bit integers, add a signed 32-bit integer and store the result as a signed 32-bit integer.

### Notes

This opcode is expected to be as efficient as basic single-precision opcodes since it utilizes the single-precision floating point multiplier.
