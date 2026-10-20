---
mnemonic: V_DIV_SCALE_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "764"
pages: [437]
source: RDNA3 ISA PDF p.437-437
---

# V_DIV_SCALE_F32 (opcode 764)

Single precision division pre-scale. S0 = Input to scale (either denominator or numerator), S1 = Denominator, S2 = Numerator. Given a numerator and denominator, this opcode appropriately scales inputs for division to avoid subnormal terms during Newton-Raphson correction method. S0 must be the same value as either S1 or S2. This opcode produces a VCC flag for post-scaling of the quotient (using V_DIV_FMAS_F32).
