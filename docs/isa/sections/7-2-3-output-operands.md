# 7.2.3. Output Operands

_(pages 69-70)_

VALU instructions typically write their results to VGPRs specified in the VDST field of the microcode word. A thread only writes a result if the associated bit in the EXEC mask is set to 1. V_CMPX instructions write the result of their comparison (one bit per thread) to the EXEC mask. Instructions producing a carry-out (integer add and subtract) write their result to VCC when used in the VOP2 form, and to an arbitrary SGPR-pair when used in the VOP3 form. When the VOP3 form is used, instructions with a floating-point result may apply an output modifier (OMOD field) that multiplies the result by: 0.5, 2.0, or 4.0. Optionally, the result can be clamped (CLAMP field) to the min and max representable range (see next section).

### 7.2.3.1. Output Operand Modifiers

Output modifiers (OMOD) apply to half, single and double floating point results only and scale the result by : 0.5, 2.0, 4.0 or do not scale. Integer and packed float 16 results ignore the omod setting. Output modifiers are not compatible with output denormals: if output denormals are enabled, then output modifiers are ignored. If output denormals are disabled, then the output modifier is applied and denormals are flushed to zero. These are not IEEE compatible: -0 is flushed to +0. Output modifiers are ignored if the IEEE mode bit is set to 1. A few opcodes force output denorms to be disabled. Output Modifiers are not supported for: • V_PERMLANE • DOT2_F16_F16 • DOT2_BF16_BF16 The clamp bit has multiple uses. For V_CMP instructions, setting the clamp bit to 1 indicates that the compare signals if a floating point exception occurs. For integer operations, it clamps the result to the largest and smallest representable value. For floating point operations, it clamps the result to the range: [0.0, 1.0]. Output Clamping: The clamp instruction bit applies to the following operations and data types: • Float clamp to [0.0, 1.0] • Signed Int [-max_int, +max_int] • Unsigned int [0, +max_int] • Bool (V_CMP) enables signaling compare The clamp bit is not supported for (ignored):

### 7.2.3.2. Wave64 Destination Restrictions

When a VALU instruction is issued from a wave64, it may issue twice as two wave32 instructions. While in most cases the programmer need not be aware of this, it does impose a prohibition on wave64 VALU instructions that both write and read the same SGPR value. Doing this may lead to unpredictable results. Specifically, the first pass of a wave64 VALU instruction may not overwrite a scalar value used by the second half.
