# 14.1. Pixel Shader Exports

_(pages 149-149)_

### Pixel Exports

Export instructions copy color data to the MRTs. Data has up to four components (R, G, B, A). Optionally, export instructions also output depth (Z) data. Every pixel shader must have at least one export instruction. The last export instruction executed must have the DONE bit set to one. The EXEC mask is applied to all exports. Only pixels with the corresponding EXEC bit set to 1 export data to the output buffer. Each export target must be exported to only once. The shader program is responsible for conversion of data from 32b to 16b for 16-bit exports. The shader program is responsible for alpha-test. All data that can affect the sample mask must be sent on the first export from the shader. This means if depth is being exported, it must be exported first. If alpha to mask is enabled, MRT0 must be exported first, unless depth is also enabled, in which case, MRT0’s alpha value must be written to the depth export’s alpha value. If alpha to mask and coverage to mask are both enabled, then the depth export’s alpha value is set to the minimum of the alpha to mask value (alpha of MRT0) and the coverage to mask value (alpha of what would have been in the depth export). If the shader can kill a pixel, it must be determined before the first export.

### Pixel Shader Dual-Source Blend

In this mode, alternating lanes (threads) hold MRT0 and MRT1, not all threads going to one MRT. There are two instructions to complete a dual-source blend export. It is required that exports to 21 and 22 be back-to- back, with no other export types in between them.
