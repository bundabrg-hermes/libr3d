# 7.2.4. Denormalized and Rounding Modes

_(pages 70-70)_

The shader program has explicit control over the rounding mode applied and the handling of denormalized inputs and results. The MODE register is set using the S_SETREG instruction; it has separate bits for controlling the behavior of single and double-precision floating-point numbers. Round and denormal modes can also be set using S_ROUND_MODE and S_DENORM_MODE which is the preferred method over using S_SETREG. 16-bit floats support denormals, infinity and NaN. Table 28. Round and Denormal Modes These mode bits do not affect rounding and denormal handling of F32 global memory atomics. DOT2_F16_F16 and DOT2_BF16_BF16 support round-to-nearest-even rounding. DOT2_F16_F16 supports denorms, and DOT2_BF16_BF16 disables all denorms.
