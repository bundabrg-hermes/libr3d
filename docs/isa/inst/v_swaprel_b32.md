---
mnemonic: V_SWAPREL_B32
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "104"
pages: [302]
source: RDNA3 ISA PDF p.302-302
---

# V_SWAPREL_B32 (opcode 104)

Swap values of two operands. The two addresses are relatively indexed using M0.

### Notes

Input and output modifiers not supported; this is an untyped operation. Example: The following instruction sequence swaps v25 and v17:
