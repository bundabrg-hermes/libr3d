# 10.9.1. Instruction definition and fields

_(pages 117-117)_

Table 53. Ray Tracing VGPR Contents Vgpr_d[4] are the destination VGPRs of the results of intersection testing. The values returned here are different depending on the type of BVH node that was fetched. For box nodes the results contain the 4 pointers of the children boxes in intersection time sorted order. For triangle BVH nodes the results contain the intersection time and triangle ID of the triangle tested. Sgpr_r[4] is the texture descriptor for the operation. The instruction is encoded with use_128bit_resource=1.

### Restrictions on image_bvh instructions

• DMASK must be set to 0xf (instruction returns all four DWORDs) • D16 must be set to 0 (16 bit return data is not supported) • R128 must be set to 1 (256 bit T#s are not supported) • UNRM must be set to 1 (only unnormalized coordinates are supported) • DIM must be set to 0 (BVH textures are 1D) • LWE must be set to 0 (LOD warn is not supported) • TFE must be set to 0 (no support for writing out the extra DWORD for the PRT hit status) • SSAMP must be set to 0 (just a placeholder, since samplers are not used by the instruction) The return order settings of the BVH ops are ignored instead they use the in-order load return queue.
