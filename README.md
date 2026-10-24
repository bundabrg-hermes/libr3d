# libr3d — a radiance kernel library for RDNA3

[radiance](https://codeberg.org/StillDeadcode/radiance) ships `libr4d`, its kernel library for
AMD RDNA4 (gfx1200/gfx1201). The engine itself starts on any AMD GPU family — its own device
code is compiled for the family-generic targets including `gfx11-generic` — but what an RDNA3
card computes depends on an installed kernel library that serves it. There is none. This repo is
that missing library: an **out-of-tree kernel plugin** for RDNA3 (gfx1100–gfx1103 /
gfx11-generic), built against radiance's ABI only, no engine fork.

Pinned to **radiance v1.11.0** (`RAD_ABI_VERSION 26`). The ABI moves with releases; rebuild this
library against the radiance version you serve.

## Status

Shipped: the bf16 model path (GEMMs, RoPE, RMSNorm, attention prefill + paged decode, sampler
ops), quantised GEMMs for **fp8 and mxfp4** including dedicated decode GEMVs, gated-delta-net
recurrent rows, and the speculative-drafter family (dflash conv/select, logit rerank, n-gram ban,
w2a8 draft-head GEMV) that makes a DFlash2 drafter place fully on-device. Every row is checked
against radiance's `libref` host oracle via `rad-kbench` across recorded real-model geometries
with OOB detection; the current verification pack runs several thousand cases green on an
RX 7900 XTX. Measured on that card, the decode GEMV bands give a **2.2x** (fp8) to **2.9x**
(mxfp4) serve-decode speedup over the fallback tile path, and a 27B-mxfp4 drafter serve goes
from 3.2 to **9.5 tok/s** once the draft pass runs on the card.

## Quick start: clone to serving

### 0. Prerequisites

- Linux with the ROCm 10.1 driver stack (`/dev/kfd`, `/dev/dri`) — only to *serve*; building is
  offline codegen and needs no RDNA3 card.
- Docker — or a native radiance v1.11.0 install (see *Without Docker* below).
- ~8 GB RAM to compile: a device-kernel translation unit takes 2–3 GB under hipcc
  (the dev image defaults to `-j2`; raise with `--build-arg JOBS=N` on bigger hosts).
- A `.rad` model to serve — radiance's `rad-convert` converts checkpoints.

### 1. Build the dev image (one-off, includes radiance)

```sh
docker build -t radiance-dev docker/
```

Takes ~30 min: it builds radiance v1.11.0 from source inside ROCm's `dev-ubuntu-24.04:10.1.0-full`
and installs it at `/opt/radiance`, so `find_package(radiance)` just resolves. To follow another
radiance release: `--build-arg RAD_REF=vX.Y.Z` — and rebuild this library against it (the ABI
must match).

### 2. Build the plugin (no card needed)

```sh
docker run --rm -v $PWD:/work radiance-dev \
  cmake -S /work -B /work/build -G Ninja -DRAD_GPU_TARGETS='gfx1101;gfx11-generic'
docker run --rm -v $PWD:/work radiance-dev cmake --build /work/build -j2
```

`gfx1100` is appended automatically if absent — the 7900 XTX needs its own chip object to reach
the WMMA rows; family-generic objects carry everything else.

The artifact is `build/radiance_home/kernels/libr3d.so`, and `build/radiance_home` is a *usable
radiance home* on its own (the build stages the empty `architectures/` sibling the loader wants).

### 3. Confirm the plugin loads (on the GPU host)

```sh
docker run --rm --device /dev/kfd --device /dev/dri \
  -v $PWD/build/radiance_home:/rhome \
  -e RADIANCE_HOME=/rhome:/opt/radiance/share/radiance \
  radiance-dev rad-info --plugins
```

`libr3d` should appear as a loaded kernel library. `$RADIANCE_HOME` is a `:`-separated search
path where the **first home wins** — prepending yours shadows an installation without copying it.

### 4. Serve a model

```sh
docker run --rm --device /dev/kfd --device /dev/dri --shm-size=4g \
  -v $PWD/build/radiance_home:/rhome -v /path/to/models:/models -p 9000:9000 \
  -e RADIANCE_HOME=/rhome:/opt/radiance/share/radiance \
  radiance-dev \
  radiance --model /models/my-model.rad \
    --placement all_vram --max-model-len 8192 --max-num-seqs 4 \
    --num-speculative-tokens 3 --host 0.0.0.0 --port 9000 --live
```

With `--placement all_vram` the engine reports each weight tensor and op band as it places them;
every band should attribute to `libr3d` with zero host-only ops. A cheap live check that the card
is doing the math: generate a few hundred tokens while watching `rocm-smi` — GPU near 100% busy,
host CPU effectively idle.

### Without Docker

Point CMake at a radiance install instead. **Host C++ must be GCC**: radiance's ABI header passes
C compound literals through `RAD_OPD`, which ROCm's clang++ rejects in C++ mode; device code still
compiles with ROCm's clang++ via `CMAKE_HIP_COMPILER`.

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_PREFIX_PATH=/opt/radiance \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DRAD_GPU_TARGETS='gfx1101;gfx11-generic'
cmake --build build -j
cmake --install build          # into the radiance home; RAD_PLUGIN_INSTALL_HOME overrides
```

## Troubleshooting: per-row kill switches

Every row family has an env var that withdraws it at the *registration* gate (a row that refused
after the resolver selected it would be fatal), so the band falls back — usually to the m64 tile,
or to the host. Use them to A/B a band or bisect a suspect kernel:

| Variable | Effect |
|---|---|
| `R3D_NO_WMMA` | prefill attention falls back to the scalar tile |
| `R3D_NO_FP8A8` | all fp8 GEMM rows withdraw (bands place on host) |
| `R3D_NO_FP8_DEC` | fp8 decode band returns to the m64 tile |
| `R3D_NO_MXFP4` | mxfp4 GEMM rows withdraw |
| `R3D_NO_FP4_DEC` | mxfp4 decode band returns to the m64 tile |
| `R3D_NO_GDN` | gated-delta-net rows withdraw |
| `R3D_NO_ANCHOR` | withdraws the recurrent rows' speculative anchor state-form |
| `R3D_NO_DFLASH` | drafter family withdraws: dflash conv/select, logit rerank, w2a8 draft GEMV |
| `R3D_NO_NGRAM_BAN` | the n-gram-ban sampler row withdraws (band resolves past the library) |
| `R3D_NO_GEMV_PF` | decode GEMVs drop the prefetch window (fp8 ships W=8; mxfp4/w2a8 opt-in) |
| `R3D_NO_SPLITK` | deep-ctx decode attention forces the single-pass tile (same row) |
| `R3D_NO_GQA` | withdraws the GQA fold arm of decode attention |

## Verifying kernels

`tools/gate.sh` wraps the verification tiers against a `rad-kbench` fixture pack
(`docs/COVERAGE.md` is the op inventory and verification plan). Kernel codegen and the non-GPU
test tiers run card-free; only kbench's GPU tier, plugin load, and serving need the card.

## Why this is tractable

- Per radiance `docs/PLUGIN.md` §10: op names are the vocabulary a library implements; a library
  with nothing for a card is *skipped, not fatal*; partial coverage resolves op-by-op.
- The bf16 model path issues only ops `libref` implements, and `libref`'s host code is the
  specification. `rad-kbench` checks each kernel against it across recorded real-model
  geometries, with OOB detection.
- `cmake/RadAddPlugin.cmake` (installed with radiance) defines the standalone build verbatim.

## Layout

    src/              plugin exports and row tables (pure host C++), kernels (.hip) beside them
    docker/           dev image (ROCm 10.1.0-full + pinned radiance install)
    tools/            gate/kbench wrappers, probes, standalone tests
    docs/COVERAGE.md  op inventory, MVP target set, verification plan

## License

Apache License 2.0 — see [LICENSE](LICENSE). This matches radiance's own license, so
headers, build glue, and code can be shared between the two without relicensing.

    Copyright 2026 bundabrg-hermes

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
