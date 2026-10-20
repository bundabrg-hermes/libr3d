# libr3d -- a radiance kernel library for RDNA3, standalone against an installed radiance.
#
# The pattern is libr4d's, copied from cmake/RadAddPlugin.cmake's documented contract: inside
# radiance's tree the parent sets everything; standalone we find the installed radiance, whose
# radiance-config.cmake imports radiance::abi and includes rad_add_plugin.

if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(libr3d LANGUAGES CXX HIP)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

# libr3d serves RDNA3 only; narrow the build's GPU targets to the gfx11 family. Built for
# anything else it is not built (mirrors libr4d's "^gfx12" gate).
rad_plugin_gpu_targets(R3D_GPU_TARGETS "^gfx11")
if(NOT R3D_GPU_TARGETS)
  message(STATUS
    "libr3d: RDNA3 (gfx11xx) only, and this build targets ${RAD_GPU_TARGETS} -- not built")
  return()
endif()

# The WMMA prefill tile is chip-gated (rocWMMA rejects family-generic targets; see the TU
# header), and the engine picks code objects chip-first-then-family -- so without a gfx1100
# object the 7900 XTX would take the gfx11-generic objects and never reach the WMMA rows.
# Radiance's default target list is gfx1101+generic; the test card is gfx1100. Generic stays
# in the list: it carries every other row (and the m64 refusal stub) onto chips we did not
# put an object for.
list(FIND R3D_GPU_TARGETS "gfx1100" R3D_HAVE_1100)
if(R3D_HAVE_1100 EQUAL -1)
  list(APPEND R3D_GPU_TARGETS gfx1100)
endif()

# Per-unit device flags, the libr4d mechanism (set_source_files_properties COMPILE_OPTIONS):
#   * -ffp-contract=off is a correctness flag in this project (rad_plugin.h: the oracle forbids
#     FMA contraction);
#   * -mno-wavefrontsize64 pins the wave32 ABI the project assumes. ROCm 10.1 already defaults
#     gfx1101 and gfx11-generic to wave32 (verified: their ISA has no ACCVGPR-joined wave64
#     shuffles); pinned anyway so no future default change silently mis-sizes the block reduce.
set_source_files_properties(src/r3d_rmsnorm.hip PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")
set_source_files_properties(src/r3d_embed.hip PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")
set_source_files_properties(src/r3d_kvstore.hip PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")
set_source_files_properties(src/r3d_actmul.hip PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")
set_source_files_properties(src/r3d_cast.hip PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")
set_source_files_properties(src/r3d_rope.hip PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")
set_source_files_properties(src/r3d_gemm_bf16.hip PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")
set_source_files_properties(src/r3d_gemm_bf16_m64.hip PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")
set_source_files_properties(src/r3d_attn_paged.hip PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")
set_source_files_properties(src/r3d_attn_paged_wmma.hip PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")
set_source_files_properties(src/r3d_attn_paged_decode.hip PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")
set_source_files_properties(src/r3d_sample.hip PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")
set_source_files_properties(src/r3d_band.hip PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")
set_source_files_properties(src/r3d_more.hip PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")
set_source_files_properties(src/r3d_gemm_mxfp4.hip PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")
set_source_files_properties(src/r3d_gdn.hip PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")

rad_add_plugin(libr3d kernels
  SOURCES     src/r3d_rows.cpp
  HIP_SOURCES src/r3d_rmsnorm.hip src/r3d_embed.hip src/r3d_kvstore.hip
              src/r3d_actmul.hip src/r3d_cast.hip src/r3d_rope.hip
              src/r3d_gemm_bf16.hip src/r3d_gemm_bf16_m64.hip src/r3d_gemm_mxfp4.hip
              src/r3d_attn_paged.hip
              src/r3d_attn_paged_wmma.hip src/r3d_attn_paged_decode.hip
              src/r3d_sample.hip
              src/r3d_band.hip src/r3d_more.hip src/r3d_gdn.hip
  GPU_TARGETS ${R3D_GPU_TARGETS}
  INCLUDES    ${CMAKE_CURRENT_SOURCE_DIR}/src)

# rad_add_plugin stages ${CMAKE_BINARY_DIR}/radiance_home/kernels/. The engine's search path
# demands a sibling architectures/ directory to exist (loader: "plugin directory
# '<home>/architectures': No such file or directory"), which a real $RADIANCE_HOME always has.
# We ship no architecture plugins, so an empty one makes the build output directly usable as
# --home for rad-info.
add_custom_command(TARGET libr3d POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "$<TARGET_FILE_DIR:libr3d>/../architectures")

# Later phases add HIP_SOURCES units beside their entry points, one translation unit per
# kernel geometry, as libr4d does.
