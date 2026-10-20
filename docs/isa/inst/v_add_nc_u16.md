---
mnemonic: V_ADD_NC_U16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "771"
pages: [440]
source: RDNA3 ISA PDF p.440-440
---

# V_ADD_NC_U16 (opcode 771)

Add two unsigned inputs and store the result into a vector register. No carry-in or carry-out support.

### Notes

Supports saturation (unsigned 16-bit integer domain).
