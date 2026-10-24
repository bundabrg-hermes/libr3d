#!/usr/bin/env bash
# Gate runner for libr3d on rdna3-dev (the only box with an RDNA3 card). Records the EXACT gate
# command lines, so a session never reconstructs them from memory. Run from the host:
#
#   ssh rdna3-dev 'bash ~/projects/libr3d/libr3d/tools/gate.sh op gdn_chunk_scan'
#   ssh rdna3-dev 'bash ~/projects/libr3d/libr3d/tools/gate.sh stock'
#   ssh rdna3-dev 'bash ~/projects/libr3d/libr3d/tools/gate.sh full'
#
# `op <name> <report-stem>`: the attribution gate -- kbench home staged with ONLY the
# oracle + libr3d (a row that "passes" while libavx's sibling answered is a phantom), over
# the 27B nodraft pack, one op. `stock`: the stock kernels.rkb regression, all libraries.
# `full`: the full mxfp4 pack, all libraries. Homes live under /scratch so libref's result
# cache survives between runs. Report for `op` lands at /scratch/<stem>.md (default
# gate_<op>.md); stock/full land at gate_kernels3.md / gate_full3.md unless $2 names one.
#
# Prereqs (ledger-pinned paths): the checkout at ~/projects/libr3d/libr3d, the build dir
# build13 built (`docker run --rm -v ~/projects/libr3d/libr3d:/work -w /work
# radiance-dev:v1.8.1 cmake --build /work/build13 -j2` -- no --device: a compile is
# card-free), and ~/scratch/r3d holding mxfp4.rkb + mxfp4_nodraft.rkb (mounted as
# /scratch). NEVER point --home or the build at the mounted build dir of the live
# serve (spec9g mounts build13 since issue #5): gates are safe (they stage copies
# from a read-only mount), but a rebuild of the serve's build dir means the NEXT
# serve restart silently adopts it -- gate first, or stage into the next build
# dir. radiance-co-tenant is production -- this script touches neither.
set -euo pipefail

REPO="$HOME/projects/libr3d/libr3d"
SCRATCH="$HOME/scratch/r3d"
IMG="${R3D_IMG:-radiance-dev:v1.11.0}"
BUILD="${R3D_BUILD:-build13}"   # pin the staged build dir; bump only after gates pass on it
DEVS=(--device /dev/kfd --device /dev/dri --group-add video)
# ROCr index of the gfx1100 XTX on this box. NEVER rely on device 0: after the slot
# move the enumeration is [gfx1201, gfx1100, gfx1201], and kbench on device 0 silently
# drops libr3d's 49 gfx11 code objects and reports a phantom all-libref pass (2026-10-23).
# Re-verify with: docker run --rm "${DEVS[@]}" radiance-dev:v1.11.0 rad-kbench --fixture ...
# (the 'device:' line lists indices) or rocm-smi --showproductname. Override: R3D_ROCR=<idx>.
ROCR="${R3D_ROCR:-1}"

stage_and_run() {  # stage_and_run <home-subdir> <libs...> -- <kbench args...>
    local home=$1; shift
    local libs=(); while [ "$1" != "--" ]; do libs+=("$1"); shift; done; shift
    local envs=(-e ROCR_VISIBLE_DEVICES="$ROCR")
    [ -n "${R3D_FP4_DOT2:-}" ] && envs+=(-e R3D_FP4_DOT2="$R3D_FP4_DOT2")
    [ -n "${R3D_NO_GEMV_PF:-}" ] && envs+=(-e R3D_NO_GEMV_PF="$R3D_NO_GEMV_PF")
    [ -n "${R3D_GEMV_PF_W:-}" ] && envs+=(-e R3D_GEMV_PF_W="$R3D_GEMV_PF_W")
    [ -n "${R3D_FP4_GEMV_PF:-}" ] && envs+=(-e R3D_FP4_GEMV_PF="$R3D_FP4_GEMV_PF")
    [ -n "${R3D_FP4_CHAIN:-}" ] && envs+=(-e R3D_FP4_CHAIN="$R3D_FP4_CHAIN")
    [ -n "${R3D_FP4_RP:-}" ] && envs+=(-e R3D_FP4_RP="$R3D_FP4_RP")
    [ -n "${R3D_NO_DFLASH:-}" ] && envs+=(-e R3D_NO_DFLASH="$R3D_NO_DFLASH")
    [ -n "${R3D_DEBUG_DFLASH:-}" ] && envs+=(-e R3D_DEBUG_DFLASH="$R3D_DEBUG_DFLASH")
    local out rc; out=$(mktemp); rc=0
    docker run --rm "${DEVS[@]}" "${envs[@]}" --shm-size=4g -v "$SCRATCH":/scratch -v "$REPO/$BUILD":/build:ro \
        --entrypoint /bin/bash "$IMG" -c "
            mkdir -p /scratch/$home/kernels /scratch/$home/architectures
            rm -f /scratch/$home/kernels/*.so
            cp /build/radiance_home/kernels/libr3d.so /scratch/$home/kernels/
            for l in ${libs[*]}; do
                [ \"\$l\" = libr3d.so ] || cp /opt/radiance/share/radiance/kernels/\$l /scratch/$home/kernels/
            done
            rad-kbench --home /scratch/$home \"\$@\"
        " _ "$@" > "$out" 2>&1 || rc=$?
    # HARD assertion: kbench must have seen ONLY the XTX as its device 0. A gate that
    # ran anywhere else is a phantom (libr3d rows silently excluded), never a result.
    if ! grep -q "1 device(s) \[0:gfx1100\]" "$out"; then
        echo "FATAL: kbench did not run on the gfx1100 alone (R3D_ROCR=$ROCR is wrong?)." >&2
        grep -m2 -E "device:|no device code" "$out" >&2
        rm -f "$out"; return 3
    fi
    cat "$out"; rm -f "$out"
    return $rc
}

tee_log() {  # tee_log <stem>: console + $SCRATCH/<stem>.log (the host name of /scratch)
    mkdir -p "$SCRATCH"; tee "$SCRATCH/$1.log"
}

case "${1:-}" in
    op)
        op=${2:?usage: gate.sh op <op-name> [report-stem]}
        stem=${3:-gate_$op}
        fix="${R3D_FIXTURE:-/scratch/mxfp4_nodraft.rkb}"
        stage_and_run home_attr libr3d.so libref.so -- \
            --fixture "$fix" --op "$op" --report "/scratch/$stem.md" \
            | tee_log "$stem"
        ;;
    stock)
        stem=${2:-gate_kernels3}
        stage_and_run home_full libr3d.so libref.so libavx.so -- \
            --fixture /opt/radiance/share/radiance/kernels.rkb --report "/scratch/$stem.md" \
            | tee_log "$stem"
        ;;
    full)
        stem=${2:-gate_full3}
        stage_and_run home_full libr3d.so libref.so libavx.so -- \
            --fixture /scratch/mxfp4.rkb --report "/scratch/$stem.md" \
            | tee_log "$stem"
        ;;
    fp8)
        # issue #4 pack: fp8_9b rows attributed libr3d-vs-libref only (no libavx
        # siblings). The #7 PF arms are selected with R3D_NO_GEMV_PF=1 / R3D_GEMV_PF_W
        # in the caller's environment.
        stem=${2:-gate_fp8}
        stage_and_run home_attr8 libr3d.so libref.so -- \
            --fixture /scratch/fp8_9b.rkb --op gemm_nt_q --report "/scratch/$stem.md" \
            | tee_log "$stem"
        ;;
    *) echo "usage: gate.sh op <op> [stem] | stock [stem] | full [stem] | fp8 [stem]" >&2; exit 2 ;;
esac
