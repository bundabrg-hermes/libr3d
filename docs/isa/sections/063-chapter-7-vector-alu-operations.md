# Chapter 7. Vector ALU Operations

_(pages 63-63)_

Vector ALU instructions (VALU) perform an arithmetic or logical operations on data for each of 32 or 64 threads and write results back to VGPRs, SGPRs or the EXEC mask. Parameter interpolation is a two step process involving an LDS instruction followed by a VALU instruction and is described in: Parameter Interpolation Vector ALU (VALU) instructions control the SIMD32’s math unit and operate on 32 work-items of data at a time. Each instruction may take input from either VGPRs, SGPRs or constants and typically returns results to VGPRs. Mask results and carry-out are returned to SGPRs. The ALU provides operations that work on 16, 32 and 64-bit data of both integer and float types. The ALU also supports "packed" data types that pack 2 16-bit values into one VGPR, or 4 8-bit values into a VGPR.
