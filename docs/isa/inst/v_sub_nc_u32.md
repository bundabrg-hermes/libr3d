---
mnemonic: V_SUB_NC_U32
section: "16.7. VOP2 Instructions"
format: VOP2
opcode: "38"
pages: [267]
source: RDNA3 ISA PDF p.267-267
---

# V_SUB_NC_U32 (opcode 38)

Subtract the second unsigned input from the first input and store the result into a vector register. No carry-in or carry-out support.

### Notes

Supports saturation (unsigned 32-bit integer domain).
