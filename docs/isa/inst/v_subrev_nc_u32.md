---
mnemonic: V_SUBREV_NC_U32
section: "16.7. VOP2 Instructions"
format: VOP2
opcode: "39"
pages: [268]
source: RDNA3 ISA PDF p.268-268
---

# V_SUBREV_NC_U32 (opcode 39)

Subtract the first unsigned input from the second input and store the result into a vector register. No carry-in or carry-out support.

### Notes

Supports saturation (unsigned 32-bit integer domain).
