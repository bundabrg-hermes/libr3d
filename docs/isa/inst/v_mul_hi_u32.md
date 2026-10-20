---
mnemonic: V_MUL_HI_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "813"
pages: [449]
source: RDNA3 ISA PDF p.449-449
---

# V_MUL_HI_U32 (opcode 813)

Multiply two unsigned integers and store the high 32 bits of the result.

### Notes

To multiply integers with small magnitudes consider V_MUL_HI_U32_U24, which is intended to be a more efficient implementation.
