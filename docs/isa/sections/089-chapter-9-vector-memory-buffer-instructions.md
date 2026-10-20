# Chapter 9. Vector Memory Buffer Instructions

_(pages 89-89)_

Vector-memory (VM) buffer operations transfer data between the VGPRs and buffer objects in memory through the texture cache (TC). Vector means that one or more piece of data is transferred uniquely for every thread in the wave, in contrast to scalar memory loads that transfer only one value that is shared by all threads in the wave. The instruction defines which VGPR(s) supply the addresses for the operation, which VGPRs supply or receive data from the operation, and a series of SGPRs that contain the memory buffer descriptor (V#). Buffer atomics have the option of returning the pre-op memory value to VGPRs. Examples of buffer objects are vertex buffers, raw buffers, stream-out buffers, and structured buffers. Buffer objects support both homogeneous and heterogeneous data, but no filtering of load-data (no samplers). Buffer instructions are divided into two groups:

### MUBUF: Untyped buffer objects

• Data format is specified in the resource constant. • Load, store, atomic operations, with or without data format conversion.

### MTBUF: Typed buffer objects

• Data format is specified in the instruction. • The only operations are Load and Store, both with data format conversion. All buffer operations use a buffer resource constant (V#) that is a 128-bit value in SGPRs. This constant is sent to the texture cache when the instruction is executed. This constant defines the address and characteristics of the buffer in memory. Typically, these constants are fetched from memory using scalar memory loads prior to executing VM instructions, but these constants also can be generated within the shader. Memory operations of different types (loads, stores) can complete out of order with respect to each other.

### Simplified view of buffer addressing

The equation below shows how the memory address is calculated for a buffer access: Memory instructions return MEMVIOL for any misaligned access when the alignment mode does not allow it.
