# 10.4. VGPR Usage

_(pages 110-112)_

Address: The address consists of up to 5 parts: { offset } { bias } { z-compare } { derivative } { body } These are all packed into consecutive VGPRs, (may be non-consecutive if "NSA" is used), and can consist of up to 12 values. • Offset: SAMPLE*O*, GATHER*O* 1 DWORD of 'offset_xyz' . The offsets are 6-bit signed integers: X=[5:0], Y=[13:8], Z=[21:16] • Bias: SAMPLE*B*, GATHER*B*. 1 DWORD float. • Z-compare: SAMPLE*C*, GATHER*C*. 1 DWORD. • Derivatives (SAMPLE_D): 2,4 or 6 DWORDS - these packed 1 DWORD per derivative as shown below (F32). • Body: One to four DWORDs, as defined by the table: Image Opcodes with a Sampler Address components are X,Y,Z,W with X in VGPR[M], Y in VGPR[M]+1, etc. The number of components in "body" is the value of the ACNT field in the table, plus one. Address components are X,Y,Z,W with X in VGPR[M], Y in VGPR[M]+1, etc. Note: Bias and Derivatives are mutually exclusive - the shader can use one or the other, but not both. 32-bit derivatives: 16-bit derivatives: The "A16" instruction bit specifies that address components are 16 bits instead of the usual 32 bits. Data : data is stored from or returned to 1-4 consecutive VGPRs. The amount of data loaded or stored is completely determined by the DMASK field of the instruction.

### Loads

DMASK specifies which elements of the resource are returned to consecutive VGPRs. The texture system loads data from memory and based on the data format expands it to a canonical RGBA form, filling in values for missing components based on T#.dst_sel. Then DMASK is applied and only those components selected are returned to the shader.

### Stores

When writing an image object, it is only possible to write an entire element (all components) - not only individual components. The components come from consecutive VGPRs and the texture system fill in the value zero for any missing components of the image’s data format, and ignore any values that are not part of the stored data format. For example if the DMASK=1001, the shader sends Red from VGPR_N and Alpha from VGPR_N+1 to the texture unit. If the image object is RGB, the texel is overwritten with Red from the VGPR_N, Green and Blue set to zero, and Alpha from the shader ignored. For D16=1, the DMASK has 1 bit set per 16-bits of data to be written from VGPRs to memory. The position of the bits in DMASK is irrelevant, only the number of bits set to 1.

### "D16" instructions

Load and store instructions also come in a "d16" variant. For stores, each 32bit VGPR holds two 16bit data elements that are passed to the texture unit which in turn, converts to the texture format before writing to memory. For loads, data returned from the texture unit is converted to 16 bits and a pair of data are stored in each 32bit VGPR (LSBs first, then MSBs). If there is only one component, the data goes into the lower half of the VGPR unless the "HI" instruction variant is used in which case the high-half of the VGPR is loaded with data.

### Atomics

Image atomic operations are supported only on 32- and 64-bit-per-pixel surfaces. The surface data format is specified in the resource constant. Atomic operations treat the element as a single component of 32- or 64- bits. For atomic operations, DMASK is set to the number of VGPRs (DWORDs) to send to the texture unit. DMASK legal values for atomic image operations: All other values of DMASK are illegal. • 0x1 = 32bit atomics except cmpswap • 0x3 = 32bit atomic cmpswap • 0x3 = 64bit atomics except cmpswap • 0xf = 64bit atomic cmpswap • Atomics with Return: Data is read out of the VGPR(s), starting at VDATA, to supply to the atomic operation. If the atomic returns a value to VGPRs, that data is returned to those same VGPRs starting at VDATA. The DMASK must be compatible with the resource’s data format.

### Denormals in Floats

Sample ops flush denormals, and loads do not modify denormals.
