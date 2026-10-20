# 2.3. Work-groups

_(pages 20-21)_

A work-group is a collection of waves which can share data through LDS and can synchronize at a barrier. Waves in a work-group are all issued to the same WGP but can run on any of the 4 SIMD32’s and can share data through LDS. The WGP supports up to 32 work-groups with a maximum of 1024 work-items per work-group. Waves in a work-group may share up to 64kB of LDS space. Work-groups consisting of a single wave do not count against the limit of 32. They do not allocate a barrier resource, and barrier ops are treated as S_NOP. Each work-group or wave can operate in one of two modes, selectable per draw/dispatch at wave-create time:

### CU mode

In this mode, the LDS is effectively split into a separate upper and lower LDS, each serving two SIMD32’s. Waves are allocated LDS space within the half of LDS which is associated with the SIMD the wave is running on. For work-groups, all waves are assigned to the pair of SIMD32’s. This mode may provide faster operation since both halves run in parallel, but limits data sharing (upper waves cannot read data in the lower half of LDS and vice versa). When in CU mode, all waves in the work-group are resident within the same CU.

### WGP mode

In this mode, the LDS is one large contiguous memory that all waves on the WGP can access. In WGP mode, waves of a work-group may be distributed across both CU’s (all 4 SIMD32’s) in the WGP. LDS_PARAM_LOAD and LDS_DIRECT_LOAD are not supported in WGP mode. The WGP (and LDS) can simultaneously have some waves running in WGP mode and other waves in CU mode running. A barrier is a synchronization primitive which makes each wave reach a given point in the shader before any wave proceeds.
