# 9.3.2. LOAD/STORE_FORMAT and DATA-FORMAT mismatches

_(pages 96-96)_

The "format" instructions specify a number of elements (x, xy, xyz or xyzw) and this could mismatch with the number of elements in the data format specified in the instruction’s or resource’s data-format field. When that happens. • buffer_load_format_x and dfmt is "32_32_32_32" : load 4 DWORDs from memory, but only load first into the shader • buffer_store_format_x and dfmt is "32_32_32_32" : stores 4 DWORDs to memory based on dst_sel • buffer_load_format_xyzw and dfmt is "32" : load 1 DWORD from memory, return 4 to shader (dst_sel) • buffer_store_format_xyzw and dfmt is "32" : store 1 DWORD (X) to memory, ignore YZW.
