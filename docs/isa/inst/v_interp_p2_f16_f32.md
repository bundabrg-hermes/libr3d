---
mnemonic: V_INTERP_P2_F16_F32
section: "16.13. VINTERP Instructions"
format: VINTERP
opcode: "3"
pages: [508]
source: RDNA3 ISA PDF p.508-508
---

# V_INTERP_P2_F16_F32 (opcode 3)

Parameter interpolation, second pass.

### Notes

Performs a hybrid 16/32-bit multiply-add operation using fixed DPP8 settings. S0 refers to a VGPR previously loaded with LDS_PARAM_LOAD that contains packed interpolation data. S1 is the I/J coordinate. S2 is the result of a previous V_INTERP_P10_F16_F32 instruction. S0 uses a fixed DPP8 lane select of {2,2,2,2,6,6,6,6}. OPSEL is allowed for D and S0 to specify which half of the register to write to/read from. Note that the I/J coordinate is 32-bit and the accumulator input is also 32-bit.
