---
mnemonic: V_INTERP_P10_F32
section: "16.13. VINTERP Instructions"
format: VINTERP
opcode: "0"
pages: [507]
source: RDNA3 ISA PDF p.507-507
---

# V_INTERP_P10_F32 (opcode 0)

Parameter interpolation, first pass.

### Notes

Performs a V_FMA_F32 operation using fixed DPP8 settings. S0 and S2 refer to a VGPR previously loaded with LDS_PARAM_LOAD that contains packed interpolation data. S1 is the I/J coordinate. S0 uses a fixed DPP8 lane select of {1,1,1,1,5,5,5,5}. S2 uses a fixed DPP8 lane select of {0,0,0,0,4,4,4,4}. Example usage:
