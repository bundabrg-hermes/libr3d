# 8.2. Dependency Checking

_(pages 87-87)_

Scalar memory loads can return data out-of-order from how they were issued; they can return partial results at different times when the load crosses two cache lines. The shader program uses the LGKMcnt counter to determine when the data has been returned to the SDST SGPRs. This is done as follows. • LGKMcnt is incremented by 1 for every fetch of a single DWORD, or cache invalidates. • LGKMcnt is incremented by 2 for every fetch of two or more DWORDs. • LGKMcnt is decremented by an equal amount when each instruction completes. Because the instructions can return out-of-order, the only sensible way to use this counter is to implement "S_WAITCNT LGKMcnt 0"; this imposes a wait for all data to return from previous SMEMs before continuing. Cache invalidate instructions are not known to have completed until the shader waits for LGKMcnt==0.
