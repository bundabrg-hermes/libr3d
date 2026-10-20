---
mnemonic: V_XAD_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "581"
pages: [427]
source: RDNA3 ISA PDF p.427-427
---

# V_XAD_U32 (opcode 581)

Calculate bitwise XOR of the first two vector inputs, then add the third vector input to the intermediate result, then store the result into a vector register.

### Notes

No carryin/carryout and no saturation. This opcode is designed to help accelerate the SHA256 hash algorithm.
