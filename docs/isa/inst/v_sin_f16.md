---
mnemonic: V_SIN_F16
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "96"
pages: [299]
source: RDNA3 ISA PDF p.299-299
---

# V_SIN_F16 (opcode 96)

Calculate the trigonometric sine of an FP16 float value using IEEE rules and store the result into a vector register. The operand is calculated by scaling the vector input by 2 PI.

### Notes

Denormals are supported. Full range input is supported. Functional examples:
