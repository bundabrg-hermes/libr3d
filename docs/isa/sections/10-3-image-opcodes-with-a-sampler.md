# 10.3. Image Opcodes with a Sampler

_(pages 109-110)_

Opcodes with a sampler: all VGPR address values are taken as FLOAT except for Texel-offset which are UINT. For cubemaps, face_id = slice * 8 + face. (Note that the "*8" differs from the non-sampler case which is "*6"). Certain sample and gather opcodes require additional values from VGPRs beyond what is shown in the table below. These values are: offset, bias, z-compare and gradients. Please see the next section for details. MSAA surfaces do not support sample or gather4 operations. The table below lists and briefly describes the legal suffixes for image instructions: Table 49. Sample Instruction Suffix Key
