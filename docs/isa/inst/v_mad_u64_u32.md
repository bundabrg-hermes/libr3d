---
mnemonic: V_MAD_U64_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "766"
pages: [439]
source: RDNA3 ISA PDF p.439-439
---

# V_MAD_U64_U32 (opcode 766)

Multiply two unsigned integer inputs, add a third unsigned integer input, store the result into a 64-bit vector register and store the overflow/carryout into a scalar mask register.

### Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair.
