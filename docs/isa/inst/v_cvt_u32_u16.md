---
mnemonic: V_CVT_U32_U16
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "107"
pages: [303]
source: RDNA3 ISA PDF p.303-303
---

# V_CVT_U32_U16 (opcode 107)

Convert from an 16-bit unsigned integer to a 32-bit unsigned integer, zero extending as needed.

### Notes

To convert in the other direction (from 32-bit to 16-bit integer) use V_MOV_B16.
