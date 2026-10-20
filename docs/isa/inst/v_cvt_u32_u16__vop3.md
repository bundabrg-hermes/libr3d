---
mnemonic: V_CVT_U32_U16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "491"
pages: [401]
source: RDNA3 ISA PDF p.401-401
---

# V_CVT_U32_U16 (opcode 491)

Convert from an 16-bit unsigned integer to a 32-bit unsigned integer, zero extending as needed.

### Notes

To convert in the other direction (from 32-bit to 16-bit integer) use V_MOV_B16.
