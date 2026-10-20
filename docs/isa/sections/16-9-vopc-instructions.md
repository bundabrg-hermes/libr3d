# 16.9. VOPC Instructions

_(family intro + opcode table, pages 304-359)_

The bitfield map for VOPC is: Compare instructions perform the same compare operation on each lane (work-Item or thread) using that lane’s private data, and producing a 1 bit result per lane into VCC or EXEC. Instructions in this format may use a 32-bit literal constant that occurs immediately after the instruction. Most compare instructions fall into one of two categories: • Those which can use one of 16 compare operations (floating point types). "{COMPF}" • Those which can use one of 8 compare operations (integer types). "{COMPI}" The opcode number is such that for these the opcode number can be calculated from a base opcode number for the data type, plus an offset for the specific compare operation. Table 112. Float Compare Operations Table 113. Instructions with Sixteen Compare Operations Table 114. Integer Compare Operations Table 115. Instructions with Eight Compare Operations
