---
mnemonic: S_VERSION
section: "16.2. SOPK Instructions"
format: SOPK
opcode: "1"
pages: [209]
source: RDNA3 ISA PDF p.209-209
---

# S_VERSION (opcode 1)

Do nothing. This opcode is used to specify the microcode version for tools that interpret shader microcode. Argument is ignored by hardware. This opcode is not designed for inserting wait states as the next instruction may issue in the same cycle. Do not use this opcode to resolve wait state hazards, use S_NOP instead. This opcode may also be used to validate microcode is running with the correct compatibility settings in drivers and functional models that support multiple generations. We strongly encourage this opcode be included at the top of every shader block to simplify debug and catch configuration errors. This opcode must appear in the first 16 bytes of a block of shader code in order to be recognized by external tools and functional models. Avoid placing opcodes > 32 bits or encodings that are not available in all versions of the microcode before the S_VERSION opcode. If this opcode is absent then tools are allowed to make an educated guess of the microcode version using cues from the environment; the guess may be incorrect and lead to an invalid decode. It is highly recommended that this be the first opcode of a shader block except for trap handlers, where it should be the second opcode (allowing the first opcode to be a 32-bit branch to accommodate context switch). SIMM16[7:0] specifies the microcode version. SIMM16[15:8] must be set to zero.
