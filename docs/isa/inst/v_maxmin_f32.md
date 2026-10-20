---
mnemonic: V_MAXMIN_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "606"
pages: [434]
source: RDNA3 ISA PDF p.434-434
---

# V_MAXMIN_F32 (opcode 606)

Compute maximum of first two operands, followed by minimum of that result and the third operand. This instruction can emulate an API-level "clamp"; unlike MED3 this correctly handles the case where the clamp's maxBound < minBound.

### Notes

Support input denorm control, allow output denorm value. Exceptions are supported. Note: +0.0 > -0.0 is true.
