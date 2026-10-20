---
mnemonic: V_FRACT_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "479"
pages: [398, 399]
source: RDNA3 ISA PDF p.398-399
---

# V_FRACT_F16 (opcode 479)

Compute the fractional portion of an FP16 float input and store the result in floating point format into a vector register.

### Notes

0.5ULP accuracy, denormals are accepted. This is intended to comply with the DX specification of fract where the function behaves like an extension of integer modulus; be aware this may differ from how fract() is defined in other domains. For example: fract(- 1.2) = 0.8 in DX.
