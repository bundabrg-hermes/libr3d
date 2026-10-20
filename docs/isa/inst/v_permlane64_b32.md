---
mnemonic: V_PERMLANE64_B32
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "103"
pages: [301, 302]
source: RDNA3 ISA PDF p.301-302
---

# V_PERMLANE64_B32 (opcode 103)

Perform a specific permutation across lanes where the high half and low half of a wave64 are swapped. Performs no operation in wave32 mode.

### Notes

In wave32 mode this opcode is translated to V_NOP and performs no writes. In wave64 the EXEC mask of the destination lane is used as the read mask for the alternate lane; as a result this opcode may read values from disabled lanes. The source must be a VGPR and SVGPRs are not allowed for this opcode. ABS, NEG and OMOD modifiers should all be zeroed for this instruction.
