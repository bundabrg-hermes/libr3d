---
mnemonic: V_READFIRSTLANE_B32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "386"
pages: [375]
source: RDNA3 ISA PDF p.375-375
---

# V_READFIRSTLANE_B32 (opcode 386)

Read the scalar value in the lowest active lane of the input vector register and store it into a scalar register.

### Notes

Overrides EXEC mask for the VGPR read. Input and output modifiers not supported; this is an untyped operation.
