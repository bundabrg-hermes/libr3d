---
mnemonic: V_ADD_NC_I32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "806"
pages: [446]
source: RDNA3 ISA PDF p.446-446
---

# V_ADD_NC_I32 (opcode 806)

Add two signed inputs and store the result into a vector register. No carry-in or carry-out support.

### Notes

Supports saturation (signed 32-bit integer domain).
