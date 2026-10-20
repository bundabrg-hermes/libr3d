# 9.2. VGPR Usage

_(pages 93-94)_

VGPRs supply address and store-data, and they can be the destination for return data.

### Address

Zero, one or two VGPRs are used, depending on the index-enable (IDXEN) and offset-enable (OFFEN) in the instruction word. These are unsigned ints. For 64-bit addresses the LSBs are in VGPRn and the MSBs are in VGPRn+1. Table 40. Address VGPRs Store Data : N consecutive VGPRs, starting at VDATA. The data format specified in the instruction word’s opcode and D16 setting determines how many DWORDs the shader provides to store. Load Data : Same as stores. Data is returned to consecutive VGPRs. Load Data Format : Load data is 32 or 16 bits, based on the data format in the instruction or resource and D16. Float or normalized data is returned as floats; integer formats are returned as integers (signed or unsigned, same type as the memory storage format). Memory loads of data in memory that is 32 or 64 bits do not undergo any format conversion unless they return as 16-bit due to D16 being set to 1. Atomics with Return : Data is read out of the VGPR(s) starting at VDATA to supply to the atomic operation. If the atomic returns a value to VGPRs, that data is returned to those same VGPRs starting at VDATA. Table 41. Data format in VGPRs and Memory Where "V0" is the VDATA VGPR; V1 is the VDATA+1 VGPR, etc.
