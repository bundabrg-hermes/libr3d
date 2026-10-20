# 12.4. LDS Direct Load

_(pages 133-134)_

Direct loads are only available in LDS, not in GDS. Direct access is allowed only in CU mode, not WGP mode. The LDS_DIRECT_LOAD instruction reads a single DWORD from LDS and returns it to a VGPR, broadcasting it to all active lanes in the wave. M0 provides the address and data type. LDS_DIRECT_LOAD uses EXEC per quad, not per pixel: if any pixel in a quad is enabled then the data is written to all 4 pixels in the quad. LDS_DIRECT_LOAD uses EXPcnt to track completion. LDS_DIRECT_LOAD uses the same instruction format and fields as LDS_PARAM_LOAD. See Pixel Parameter Interpolation. Signed byte and short data is sign-extend to 32 bits before writing the result to a VGPR; unsigned byte and short data is zero-extended to 32 bits before writing to a VGPR.
