---
mnemonic: V_SUB_NC_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "294"
pages: [410]
source: RDNA3 ISA PDF p.410-410
---

# V_SUB_NC_U32 (opcode 294)

Subtract the second unsigned input from the first input and store the result into a vector register. No carry-in or carry-out support.

### Notes

Supports saturation (unsigned 32-bit integer domain).
