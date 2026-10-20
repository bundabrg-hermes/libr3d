---
mnemonic: V_DIV_FIXUP_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "551"
pages: [422]
source: RDNA3 ISA PDF p.422-422
---

# V_DIV_FIXUP_F32 (opcode 551)

Single precision division fixup. S0 = Quotient, S1 = Denominator, S2 = Numerator. Given a numerator, denominator, and quotient from a divide, this opcode detects and applies specific case numerics, touching up the quotient if necessary. This opcode also generates invalid, denorm and divide by zero exceptions caused by the division.
