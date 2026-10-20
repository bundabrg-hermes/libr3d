---
mnemonic: V_SIN_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "437"
pages: [387]
source: RDNA3 ISA PDF p.387-387
---

# V_SIN_F32 (opcode 437)

Calculate the trigonometric sine of a single-precision float value using IEEE rules and store the result into a vector register. The operand is calculated by scaling the vector input by 2 PI.

### Notes

Denormals are supported. Full range input is supported. Functional examples:
