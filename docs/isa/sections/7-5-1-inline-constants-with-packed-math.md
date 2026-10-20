# 7.5.1. Inline Constants with Packed Math

_(pages 76-76)_

Inline constants may be used with packed math, but they require the use of OPSEL. Inline constants produce a value in only the low 16-bits of the 32-bit constant value. Inline constants used with float 16-bit sources produce an F16 constant value. Without using OPSEL, only the lower half of the source would contain the constant. To use the inline constant in both halves, use OPSEL to select the lower input for both low and high sources. BF16 uses 32-bit float constants and then the BF16 operand selects the upper 16 bits of the FP32 constant (matches the definition of BF16). For the WMMA_F16_F16_16x16x16 or VOPD DOT2_F32_F16, hardware automatically selects the low 16 bits of the constant. Any packed math instructions that use data sizes less than 16 bits do not work with inline constants, other than the DOT instructions below:
