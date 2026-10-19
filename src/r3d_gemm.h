/* r3d_gemm.h -- internal hand-off between gemm entry TU and the tile TUs.
 *
 * r3d_gemm_bf16.hip owns the gemm_nt/logits_gemm entries and the M-derivation gate;
 * r3d_gemm_bf16_m64.hip (the WMMA prefill tile) exposes its launcher here. Plain C++
 * linkage within the plugin .so; not part of the plugin's export contract.
 */
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

/* Launches the 64x64 WMMA tile. Returns RAD_OK when enqueued, RAD_E_DEVICE on a launch
 * failure (the caller then falls back to the scalar tile, which was never launched). */
int r3d_gemm_nt_m64_launch(const void* A, const void* B, void* Y, float* Yf,
                           int M, int N, int K, int64_t a_pitch, int64_t b_pitch,
                           int64_t y_pitch, hipStream_t stream);
