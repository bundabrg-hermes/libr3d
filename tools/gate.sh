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
# /scratch). NEVER point --home or the build at build11: the live serve mounts it, and
# radiance-co-tenant is production -- this script touches neither.
set -euo pipefail

REPO=/home/devuser/projects/libr3d/libr3d
SCRATCH=/home/devuser/scratch/r3d
IMG=radiance-dev:v1.8.1
DEVS=(--device /dev/kfd --device /dev/dri --group-add video)

stage_and_run() {  # stage_and_run <home-subdir> <libs...> -- <kbench args...>
    local home=$1; shift
    local libs=(); while [ "$1" != "--" ]; do libs+=("$1"); shift; done; shift
    docker run --rm "${DEVS[@]}" --shm-size=4g -v "$SCRATCH":/scratch -v "$REPO/build13":/build13:ro \
        --entrypoint /bin/bash "$IMG" -c "
            mkdir -p /scratch/$home/kernels /scratch/$home/architectures
            rm -f /scratch/$home/kernels/*.so
            cp /build13/radiance_home/kernels/libr3d.so /scratch/$home/kernels/
            for l in ${libs[*]}; do
                [ \"\$l\" = libr3d.so ] || cp /opt/radiance/share/radiance/kernels/\$l /scratch/$home/kernels/
            done
            rad-kbench --home /scratch/$home \"\$@\"
        " _ "$@"
}

tee_log() {  # tee_log <stem>: console + $SCRATCH/<stem>.log (the host name of /scratch)
    mkdir -p "$SCRATCH"; tee "$SCRATCH/$1.log"
}

case "${1:-}" in
    op)
        op=${2:?usage: gate.sh op <op-name> [report-stem]}
        stem=${3:-gate_$op}
        stage_and_run home_attr libr3d.so libref.so -- \
            --fixture /scratch/mxfp4_nodraft.rkb --op "$op" --report "/scratch/$stem.md" \
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
    *) echo "usage: gate.sh op <op> [stem] | stock [stem] | full [stem]" >&2; exit 2 ;;
esac
