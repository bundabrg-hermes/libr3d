# Chapter 4. Shader Instruction Set

_(pages 43-43)_

This chapter describes the shader instruction set. Instructions are divided into the following groups: • Program Flow • Scalar ALU • Scalar memory read from constant cache • Vector ALU & Parameter-Interpolate • Vector Memory read/write : ◦buffers ◦Flat, Global and Scratch ◦LDS • GDS • Misc: wait on counter, barrier, send message Instructions are encoded in various microcode formats. The formats are defined by a set of "encoding" bits (in red) that define the family of instructions and the meaning of the rest of the bits in the instruction. Not every instruction uses every field in its encoding. Fields which can specify an SGPR as a source or dest are typically set to NULL when unused; other fields are typically set to zero.
