# 2.2.2. Graphics Shaders

_(pages 19-20)_

The shader supports 3 types of graphics waves: PS, GS, and HS. Rendering modes (launch behavior): • Normal NGG - Geometry Engine (GE) sends info to wave launch hardware to init VGPRs for each element (prim) launched; GE fetches index and vertex buffer data and loads to VGPRs • Mesh shader - turns GS-launch into a CS-style launch, and wave launch hardware does unrolling into elements and generates element indices on the fly. The mesh shader program determines how to use this index value. The amplification shader decides how many mesh shader groups to launch. The mesh shader processes vertices and then primitives.
