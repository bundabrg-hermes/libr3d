---
mnemonic: V_INTERP_P10_F16_F32
section: "16.13. VINTERP Instructions"
format: VINTERP
opcode: "2"
pages: [508]
source: RDNA3 ISA PDF p.508-508
---

# V_INTERP_P10_F16_F32 (opcode 2)

Parameter interpolation, first pass.

### Notes

Performs a hybrid 16/32-bit multiply-add operation using fixed DPP8 settings. S0 and S2 refer to a VGPR previously loaded with LDS_PARAM_LOAD that contains packed interpolation data. S1 is the I/J coordinate. S0 uses a fixed DPP8 lane select of {1,1,1,1,5,5,5,5}. S2 uses a fixed DPP8 lane select of {0,0,0,0,4,4,4,4}. OPSEL is allowed for S0 and S2 to specify which half of the register to read from. Note that the I/J coordinate is 32-bit and the destination is also 32-bit.
