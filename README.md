# libr3d — a radiance kernel library for RDNA3

[radiance](https://codeberg.org/StillDeadcode/radiance) ships `libr4d`, its kernel library for
AMD RDNA4 (gfx1200/gfx1201). The engine itself starts on any AMD GPU family — its own device
code is compiled for the family-generic targets including `gfx11-generic` — but what an RDNA3
card computes depends on an installed kernel library that serves it. There is none. This repo is
that missing library: an **out-of-tree kernel plugin** for RDNA3 (gfx1100–gfx1103 /
gfx11-generic), built against radiance's ABI only, no engine fork.

Pinned to **radiance v1.8.1** (`RAD_ABI_VERSION 22`). The ABI moves with releases; rebuild this
library against the radiance version you serve.

## Status

Bootstrap phase: the repo scaffolds a loadable plugin (empty kernel table — the loader accepts
it and simply has no device rows for gfx11). The plan is in [docs/COVERAGE.md](docs/COVERAGE.md):
a bf16 minimum-viable op set checked against radiance's `libref` host oracle via `rad-kbench`,
then widening toward the fp8/quantised vocabulary.

## Why this is tractable

- Per radiance `docs/PLUGIN.md` §10: op names are the vocabulary a library implements; a library
  with nothing for a card is *skipped, not fatal*; partial coverage resolves op-by-op.
- The bf16 model path (e.g. MiniCPM5-2B: 13 model ops + ~11 sampler ops) issues only ops
  `libref` implements, and `libref`'s host code is the specification. `rad-kbench` checks each
  kernel against it across recorded real-model geometries, with OOB detection.
- `cmake/RadAddPlugin.cmake` (installed with radiance) defines the standalone build verbatim.

## Development without a card

Kernel compilation for gfx11 is offline codegen; no RDNA3 hardware is needed to build or to run
the non-GPU test tiers (`ctest -L fast`, `ctest -LE 'gpu|oot'` upstream). Only kbench's GPU tier,
plugin load confirmation (`rad-info --plugins`) and serving need the card.

Container dev environment (`docker/Dockerfile.dev`):

```sh
docker build -t radiance-dev docker/            # ROCm 10.1 + radiance v1.8.1 installed at /opt/radiance
docker run -v $PWD:/work radiance-dev \
  cmake -S /work -B /work/build -G Ninja -DRAD_GPU_TARGETS='gfx1101;gfx11-generic'
docker run -v $PWD:/work radiance-dev \
  cmake --build /work/build
```

On the RDNA3 host (rdna3-dev), run the same image with `--device /dev/kfd --device /dev/dri` and run
`rad-kbench` / `rad-info` against the card. Keeping dev and test on one image avoids toolchain
drift; note upstream's shipping image additionally pins a patched libLLVM for build
repeatability (ROCm 10.1 VOPD pairing), which the dev image does not carry — fine for
development, worth mirroring for shipped binaries.

## Layout

    src/            plugin exports and row tables (pure host C++), kernels (.hip) beside them
    docker/         dev image (ROCm 10.1.0-full + pinned radiance install)
    docs/COVERAGE.md  op inventory, MVP target set, verification plan
