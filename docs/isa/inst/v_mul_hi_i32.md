---
mnemonic: V_MUL_HI_I32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "814"
pages: [449]
source: RDNA3 ISA PDF p.449-449
---

# V_MUL_HI_I32 (opcode 814)

Multiply two signed integers and store the high 32 bits of the result.

### Notes

To multiply integers with small magnitudes consider V_MUL_HI_I32_I24, which is intended to be a more efficient implementation.
