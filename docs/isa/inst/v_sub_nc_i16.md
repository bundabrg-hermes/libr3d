---
mnemonic: V_SUB_NC_I16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "782"
pages: [442, 443]
source: RDNA3 ISA PDF p.442-443
---

# V_SUB_NC_I16 (opcode 782)

Subtract the second signed input from the first input and store the result into a vector register. No carry-in or carry-out support.

### Notes

Supports saturation (signed 16-bit integer domain).
