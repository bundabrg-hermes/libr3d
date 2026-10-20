---
mnemonic: V_SUB_NC_U16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "772"
pages: [441]
source: RDNA3 ISA PDF p.441-441
---

# V_SUB_NC_U16 (opcode 772)

Subtract the second unsigned input from the first input and store the result into a vector register. No carry-in or carry-out support.

### Notes

Supports saturation (unsigned 16-bit integer domain).
