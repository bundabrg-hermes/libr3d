---
mnemonic: V_ADD_NC_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "293"
pages: [410]
source: RDNA3 ISA PDF p.410-410
---

# V_ADD_NC_U32 (opcode 293)

Add two unsigned inputs and store the result into a vector register. No carry-in or carry-out support.

### Notes

Supports saturation (unsigned 32-bit integer domain).
