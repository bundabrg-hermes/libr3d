---
mnemonic: V_DIV_FIXUP_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "552"
pages: [423]
source: RDNA3 ISA PDF p.423-423
---

# V_DIV_FIXUP_F64 (opcode 552)

Double precision division fixup. S0 = Quotient, S1 = Denominator, S2 = Numerator. Given a numerator, denominator, and quotient from a divide, this opcode detects and applies specific case numerics, touching up the quotient if necessary. This opcode also generates invalid, denorm and divide by zero exceptions caused by the division.
