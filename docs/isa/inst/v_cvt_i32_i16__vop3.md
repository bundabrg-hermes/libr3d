---
mnemonic: V_CVT_I32_I16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "490"
pages: [401]
source: RDNA3 ISA PDF p.401-401
---

# V_CVT_I32_I16 (opcode 490)

Convert from an 16-bit signed integer to a 32-bit signed integer, sign extending as needed.

### Notes

To convert in the other direction (from 32-bit to 16-bit integer) use V_MOV_B16.
