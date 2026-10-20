# 10.9. Ray Tracing

_(pages 116-116)_

Ray Tracing support includes the following instructions: • IMAGE_BVH_INTERSECT_RAY • IMAGE_BVH64_INTERSECT_RAY These instructions receive ray data from the VGPRs and fetch BVH (Bounding Volume Hierarchy) from memory. • Box BVH nodes perform 4x Ray/Box intersection, sorts the 4 children based on intersection distance and returns the child pointers and hit status. • Triangle nodes perform 1 Ray/Triangle intersection test and returns the intersection point and triangle ID. The two instructions are identical, except that the "64" version supports a 64-bit address while the normal version supports only a 32bit address. Both instructions can use the "A16" instruction field to reduce some (but not all) of the address components to 16 bits (from 32). These addresses are: ray_dir and ray_inv_dir.
