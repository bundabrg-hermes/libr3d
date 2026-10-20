# 2.2.1. Compute Shaders

_(pages 19-19)_

Compute kernels (shaders) are generic programs that can run on the RDNA3 processor, taking data from memory, processing it, and writing results back to memory. Compute kernels are created by a dispatch, which causes the RDNA3 processors to run the kernel over all of the work-items in a 1D, 2D, or 3D grid of data. The RDNA3 processor walks through this grid and generates waves, which then run the compute kernel. Each work-item is initialized with its unique address (index) within the grid. Based on this index, the work-item computes the address of the data it is required to work on and what to do with the results.
