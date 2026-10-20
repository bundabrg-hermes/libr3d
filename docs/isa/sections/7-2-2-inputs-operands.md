# 7.2.2. Inputs Operands

_(pages 67-68)_

VALU instructions can use any of the following sources for input, subject to restrictions listed below: • VOP1, VOP2, VOPC: ◦SRC0 is 9 bits and may be a VGPR, SGPR (including TTMPs and VCC), M0, EXEC, inline or literal constant. ◦SRC1 is 8 bits and may specify only a VGPR • VOP3 : all 3 sources are 9 bits but still have restrictions: ◦Not all VOPC/1/2 instructions are available in VOP3 (only those that benefit from VOP3 encoding). • See complete operand list: VALU Instruction Operands

### 7.2.2.1. Input Operand Modifiers

The input modifiers ABS and NEG apply to floating point inputs and are undefined for any other type of input. In addition, input modifiers are supported for: V_MOV_B32, V_MOV_B16, V_MOVREL*_B32 and V_CNDMASK. ABS returns the absolute value, and NEG negates the input. Input modifiers are not supported for: • readlane, readfirstlane, writelane • integer arithmetic or bitwise operations • permlane • QSAD

### 7.2.2.2. Literal Expansion to 64 bits

Literal constants are 32-bits, but they can be used as sources that normally require 64-bit data. They are expanded to 64 bits following these rules: • 64 bit float: the lower 32-bit are padded with zero • 64-bit unsigned integer: zero extended to 64 bits • 64-bit signed integer: sign extended to 64 bits

### 7.2.2.3. Source Operand Restrictions

Not every combination of source operands that can be expressed in the microcode format is legal. This section describes the legal and illegal settings. Terminology for this section: "scalar value" = SGPR, EXEC, VCC, M0, SCC or literal constant; can be 32 or 64 bits. • Instructions may use at most two Scalar Values: SGPR, VCC, M0, EXEC, SCC, Literal • All instruction formats including VOP3 and VOP3P may use one literal constant ◦Inline constants are free (do not count against 2 scalar value limit). ◦Literals may not be used with DPP ◦It is permissible for both scalar values to be SGPRs, although VCC counts as an SGPR. ▪VCC when used implicitly counts against this limit: addci, subci, fmas, cndmask ◦64-bit shift instructions can use only one scalar value input, and can’t use the same one twice (inlines don’t count against this limit) ◦Using the same scalar value twice only counts as a single scalar value, however using the same scalar value twice, but with different sizes has specific rules and limits: ▪Using the same literal with different sizes counts as 2 scalar values, not 1. ▪S[0] and S[0:1] can be considered as 1 scalar value, but S[1] and S[0:1] count as 2. In general, these rules apply to any S[2n] and S[2n:2n+1] count as one, but S[2n+1] and S[2n:2n+1] count as 2. • SGPR source rules must be met for both passes of a wave64, bearing in mind that sources that read a mask (bit-per-lane) increment the SGPR address for the second pass, and they may not be shared with other sources.

### 7.2.2.4. OPSEL Field Restrictions

The OPSEL field (of VOP3) is usable only for a subset of VOP3 instructions, as well as VOP1/2/C instructions promoted to VOP3. Table 27. Opcodes usable with OPSEL
