---
mnemonic: V_INTERP_P2_F32
section: "16.13. VINTERP Instructions"
format: VINTERP
opcode: "1"
pages: [507]
source: RDNA3 ISA PDF p.507-507
---

# V_INTERP_P2_F32 (opcode 1)

Parameter interpolation, second pass.

### Notes

Performs a V_FMA_F32 operation using fixed DPP8 settings. S0 refers to a VGPR previously loaded with LDS_PARAM_LOAD that contains packed interpolation data. S1 is the I/J coordinate. S2 is the result of a previous V_INTERP_P10_F32 instruction. S0 uses a fixed DPP8 lane select of {2,2,2,2,6,6,6,6}.
