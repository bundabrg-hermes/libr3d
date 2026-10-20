# 10.2. Image Opcodes with No Sampler

_(pages 107-109)_

For image opcodes with no sampler, all VGPR address values are taken as uint. For cubemaps, face_id = slice * 6 + face. MSAA surfaces support only load, store and atomics; not load-mip or store-mip. The table below shows the contents of address VGPRs for the various image opcodes. • Image_Load : image_load, image_load_mip, image_load_{pck, pck_sgn, mip_pck, mip_pck_sgn} • Image_Store: image_store, image_store_mip • Image_Atomic_*: swap, cmpswap, add, sub, {u,s}{min,max}, and, or, xor, inc, dec. "ACNT" is the Address Count: the number of VGPRs that supply the "body" of the address, derived from the instruction’s DIM field and the opcode.
