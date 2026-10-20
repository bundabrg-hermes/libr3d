---
mnemonic: V_SUB_CO_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "769"
pages: [440]
source: RDNA3 ISA PDF p.440-440
---

# V_SUB_CO_U32 (opcode 769)

Subtract the second unsigned input from the first input, store the result into a vector register and store the carry-out mask into a scalar register.

### Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair. Supports saturation (unsigned 32-bit integer domain).
