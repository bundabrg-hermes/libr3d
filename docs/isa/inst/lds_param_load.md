---
mnemonic: LDS_PARAM_LOAD
section: "16.14. Parameter and Direct Load from LDS Instructions"
format: PARAM
opcode: "0"
pages: [510]
source: RDNA3 ISA PDF p.510-510
---

# LDS_PARAM_LOAD (opcode 0)

Transfer parameter data from LDS to VGPRs and expand data in LDS using the NewPrimMask (provided in M0) to place per-quad data into lanes 0-3 of each quad as follows: {P0, P10, P20, 0.0} This data may be extracted using DPP8 for interpolation operations. The V_INTERP_* instructions unpack data automatically. When loading FP16 parameters, two attributes are loaded into a single VGPR: Attribute 2*ATTR is loaded into the low 16 bits and attribute 2*ATTR+1 is loaded into the high 16 bits. This instruction runs in whole quad mode: if any pixel of a quad is active then all 4 pixels of that quad are written. This is required for interpolation instructions to have all the parameter information available for the quad.
