---
mnemonic: V_SUB_CO_CI_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "289"
pages: [409]
source: RDNA3 ISA PDF p.409-409
---

# V_SUB_CO_CI_U32 (opcode 289)

Subtract the second unsigned input from the first input, subtract a bit from the carry-in mask, store the result into a vector register and store the carry-out mask to a scalar register.

### Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair, and the VCC source comes from the SGPR-pair at S2.u. Supports saturation (unsigned 32-bit integer domain).
