---
mnemonic: S_CMP_EQ_U32
section: "16.4. SOPC Instructions"
format: SOPC
opcode: "6"
pages: [239]
source: RDNA3 ISA PDF p.239-239
---

# S_CMP_EQ_U32 (opcode 6)

Set SCC to 1 iff the first scalar input is equal to the second scalar input.

### Notes

Note that S_CMP_EQ_I32 and S_CMP_EQ_U32 are identical opcodes, but both are provided for symmetry.
