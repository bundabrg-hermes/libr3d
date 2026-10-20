---
mnemonic: V_COS_F32
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "54"
pages: [287]
source: RDNA3 ISA PDF p.287-287
---

# V_COS_F32 (opcode 54)

Calculate the trigonometric cosine of a single-precision float value using IEEE rules and store the result into a vector register. The operand is calculated by scaling the vector input by 2 PI.

### Notes

Denormals are supported. Full range input is supported. Functional examples:
