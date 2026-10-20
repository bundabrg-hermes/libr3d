# 7.7. Data Parallel Processing (DPP)

_(pages 78-79)_

Data Parallel Processing (DPP) operations allow VALU instruction to select operands from different lanes (threads) rather than just using a thread’s own data. DPP operations are indicated by the use of the inline constant: DPP8 or DPP16 in the SRC0 operand. Note that since SRC0 is set to the DPP value, the actual VGPR address for SRC0 comes from the DPP DWORD. One example of using DPP is for scan operations. A scan operation is one that computes a value per thread that is based on the values of the previous threads and possibly itself. E.g. a running sum is the sum of the values from previous threads in the vector. A reduction operation is essentially a scan that returns a single value from the highest numbered active thread. A scan operation requires that the EXEC mask to be set to all 1’s for proper operation. Unused threads (lanes) should be set to a value that does not change the result prior to the scan. There are two forms of the DPP instruction word:

### DPP8

allows arbitrary swizzling between groups of 8 lanes

### DPP16

allows a set of predefined swizzles between groups of 16 lanes DPP may be used only with: VOP1, VOP2, VOPC, VOP3 and VOP3P (but not "packed math" ops). DPP instructions incur an extra cycle of delay to execute. Table 30. Which instructions support DPP V_CMP and V_CMPX write the full mask, not a partial mask. When using DPP with V_CMP or V_CMPX and setting bound_ctrl=0, lanes that have their EXEC mask bit set to zero instead of not writing the bit, a zero bit is written. "FI" (Fetch Inactive) with DPP16 causes a lane to act as if it is active when supplying data, but the compare result for that lane is still zero for V_CMPX (V_CMPX with FI=1 does not turn on a lane that was off).
