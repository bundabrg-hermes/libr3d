---
mnemonic: V_CVT_I32_I16
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "106"
pages: [302, 303]
source: RDNA3 ISA PDF p.302-303
---

# V_CVT_I32_I16 (opcode 106)

Convert from an 16-bit signed integer to a 32-bit signed integer, sign extending as needed.

### Notes

To convert in the other direction (from 32-bit to 16-bit integer) use V_MOV_B16.
