---
mnemonic: V_ADD_CO_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "768"
pages: [439]
source: RDNA3 ISA PDF p.439-439
---

# V_ADD_CO_U32 (opcode 768)

Add two unsigned inputs, store the result into a vector register and store the carry-out mask into a scalar register.

### Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair. Supports saturation (unsigned 32-bit integer domain).
