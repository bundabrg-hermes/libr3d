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

# Per-unit device flags, the libr4d mechanism (set_source_files_properties COMPILE_OPTIONS):
#   * -ffp-contract=off is a correctness flag in this project (rad_plugin.h: the oracle forbids
#     FMA contraction);
#   * -mno-wavefrontsize64 pins the wave32 ABI the project assumes. ROCm 10.1 already defaults
#     gfx1101 and gfx11-generic to wave32 (verified: their ISA has no ACCVGPR-joined wave64
#     shuffles); pinned anyway so no future default change silently mis-sizes the block reduce.
set_source_files_properties(src/r3d_rmsnorm.hip PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-mno-wavefrontsize64")

rad_add_plugin(libr3d kernels
  SOURCES     src/r3d_rows.cpp
  HIP_SOURCES src/r3d_rmsnorm.hip
  GPU_TARGETS ${R3D_GPU_TARGETS}
  INCLUDES    ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Later phases add HIP_SOURCES units beside their entry points, one translation unit per
# kernel geometry, as libr4d does.
