# 7.5. Packed Math

_(pages 74-76)_

Packed math is a form of operation that accelerates arithmetic on two values packed into the same VGPR. It performs operations on two 16-bit values within a DWORD as if they were separate threads. For example, a packed add of V0=V1+V2 is really two separate adds: adding the low 16 bits of each DWORD and storing the result in the low 16 bits of V0, and adding the high halves and storing the result in the high 16 bits of V0. Packed math uses the instructions below and the microcode format "VOP3P". This format has OPSEL and NEG fields for both the low and high operands, and does not have ABS and OMOD. Table 29. Packed Math Opcodes:  V_FMA_MIX_* and WMMA instructions are not packed math, but perform a single MAD operation on a mixture of 16- and 32-bit inputs. They are listed here because they use the VOP3P encoding.

### VOP3P Instruction Fields

### OPSEL for MIX instructions

MIX, MIXLO and MIXHI interpret OPSEL and OPSEL_HI as three 2-bit fields, one per source operand: { OPSEL_HI[0], OPSEL[0] } controls source0; { OPSEL_HI[1], OPSEL[1] } controls source1; { OPSEL_HI[2], OPSEL[2] } controls source2. These 2-bit fields control source-selection for each of the 3 source operands: 2’b00: Src[31:0] as FP32 2’b01: Src[31:0] as FP32 2’b10: Src[15:0] as FP16 2’b11: Src[31:16] as FP16 V_WMMA…IU… and V_DOT4…IU… with NEG:: These instructions use the NEG[1:0] bits to indicate signed (0=unsigned, 1=signed) per input source instead of meaning "negate". NEG[2] should be set to zero (behavior is undefined). NEG_HI must be zero.
