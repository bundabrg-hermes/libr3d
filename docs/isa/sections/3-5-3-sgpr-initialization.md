# 3.5.3. SGPR Initialization

_(pages 37-39)_

SGPRs are initialized based on various SPI_PGM_RSRC* or COMPUTE_PGM_* register settings. Note that only the enabled values are loaded, and they are packed into consecutive SGPRs, skipping over disabled values regardless of the number of user-constants loaded. No SGPRs are skipped for alignment. The tables below show how to control which values are initialized prior to shader launch.

### 3.5.3.1. Pixel Shader (PS)

Table 8. PS SGPR Load PS_wave_index is (se_id[1:0] * GPU__GC__NUM_PACKER_PER_SE + packer_id). PS_wave_id is an index value which is incremented for every wave. There is a separate counter per packer, so the combination of { ps_wave_id, ps_wave_index } forms a unique ID for any wave on the chip. The wave-id counter wraps at SPI_PS_MAX_WAVE_ID.

### 3.5.3.2. Geometry Shader (GS)

ES and GS are launched as a combined wave, of type GS. The shader is initialized as a GS wave type, with the PC pointing to the ES shader and with GS user-SGPRs preloaded, along with a memory pointer to more GS user SGPRs. The shader executes to the ES program first, then upon completion executes the GS shader. Once the ES shader completes, it may re-use the SGPRs which contain ES user data and the GS shader address. The first 8 SGPRs are automatically initialized - no values are skipped (unused ones are written with zero). State registers: • SPI_SHADER_PGM_{LO,HI}_ES : address of the GS shader • SPI_SHADER_PGM_RSRC1: resources of combined ES + GS shader ◦GS_VGPR_COMP_CNT = # of GS VGPRs to load (2 bits) • SPI_SHADER_PGM_RSRC2: resources of combined ES + GS shader ◦VGPR_COMP_CNT = # of VGPRs to load (2 bits) ◦OC_LDS_EN • SPI_SHADER_PGM_RSRC{3,4}: resources of combined ES + GS shader Table 9. GS SGPR Load When stream-out is used, SGPR[2] must not be modified or overwritten any time before the final stream out is issued (GDS ordered count with 'done' = 1). This is because the pipeline reset sequence which hardware automatically executes reads SGPR to fabricate a GDS-ordered-count instruction and relies on this value.

### 3.5.3.3. Front End Shader (HS)

LS and HS are launched as a combined wave, of type HS. The shader is initialized as an HS wave type, with the PC pointing to the LS shader and with HS user-SGPRs preloaded, along with a memory pointer to more HS user SGPRs. The shader executes to the LS program first, then upon completion executes the HS shader. Once the LS shader completes, it may re-use the SGPRs which contain LS user data and the HS shader address. The first 8 SGPRs are automatically initialized - no values are skipped (unused ones are written with zero). Other registers: • SPI_SHADER_PGM_{LO,HI}_LS : address of the LS shader • SPI_SHADER_PGM_RSRC1: resources of combined LS + HS shader ◦LS_VGPR_COMP_CNT = # of LS VGPRs to load (2 bits) • SPI_SHADER_PGM_RSRC{2,3,4}: resources of combined LS + HS shader Table 10. HS (LS) SGPR Load

### 3.5.3.4. Compute Shader (CS)

Table 11. CS SGPR Load
