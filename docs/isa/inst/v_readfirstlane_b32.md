---
mnemonic: V_READFIRSTLANE_B32
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "2"
pages: [274, 275]
source: RDNA3 ISA PDF p.274-275
---

# V_READFIRSTLANE_B32 (opcode 2)

Read the scalar value in the lowest active lane of the input vector register and store it into a scalar register.

### Notes

Overrides EXEC mask for the VGPR read. Input and output modifiers not supported; this is an untyped operation.
