# 8.1. Microcode Encoding

_(pages 85-86)_

Scalar memory load instructions are encoded using the SMEM microcode format. The fields are described in the table below: Table 34. SMEM Encoding Field Descriptions Table 35. SMEM Instructions These instructions load 1-16 DWORDs from memory. The data in SGPRs is specified in SDATA, and the address is composed of the SBASE, OFFSET, and SOFFSET fields.
