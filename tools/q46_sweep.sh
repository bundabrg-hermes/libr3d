#!/usr/bin/env bash
# Q46 timing sweep -- CARD EXCLUSIVE (spec9g stopped, battery done, nothing else
# on the XTX). Inside radiance-dev, ROCR_VISIBLE_DEVICES=1, with the gate.sh
# hard assertion replicated (gfx1100 alone or the leg is a phantom).
# h128 serve geometry (32Q/8KV fold, rows=1) at 1024/4096/8192, ONE binary
# (build_v154), alternating arms A,B per shape per round x3, reps=15, sets=8
# (KV plane past the 96 MB Infinity Cache at >=4096; 1024 cache-fit -- regime
# labelled in the report). Plus p48 retune: h256 16Q/2KV ctx 8192 law-vs-wave-vs-f48.
set -uo pipefail
REPO=$HOME/projects/libr3d/libr3d
if [ -z "${Q46_IN_CONTAINER:-}" ]; then
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    -e ROCR_VISIBLE_DEVICES=1 -e Q46_IN_CONTAINER=1 \
    -v $REPO:/work -w /work --shm-size=4g --entrypoint /bin/bash \
    radiance-dev:v1.11.0 /work/tools/q46_sweep.sh
  exit $?
fi
cd /work
mkdir -p q46logs; LOG=q46logs/q46_sweep.log; : > $LOG
say() { echo "$@" | tee -a $LOG; }
dv=$(./q46bin/dev0); echo "$dv" | tee -a $LOG
case "$dv" in "1 device(s) [0:gfx1100]") ;; *) echo "FATAL: not gfx1100 alone -- phantom, abort" | tee -a $LOG; exit 3;; esac
export LIBR3D_SO=/work/build_v154/radiance_home/kernels/libr3d.so
say "== sweep start $(date -u +%H:%M:%S) =="
for R in 1 2 3; do
 for CTX in 1024 4096 8192; do
  for ARM in "p0:0:" "p1cu48:1:" "p1cu96:1:96"; do
    NAME=${ARM%%:*}; rest=${ARM#*:}; POL=${rest%%:*}; CU=${rest#*:}
    f=q46logs/ab_${CTX}_${NAME}_r${R}
    declare -a EV=(R3D_SPLIT_POLICY=$POL R3D_DEBUG_DECODE=1)
    [ -n "$CU" ] && EV+=(R3D_SPLIT_CU=$CU)
    env "${EV[@]}" ./q46bin/bench_attn_decode time 32 8 1 $CTX 15 8 > $f.txt 2>&1 || say "RUNFAIL $f"
    arm=$(grep -m1 -oE "split-policy arm=[a-z]+ cu=[0-9]+" $f.txt)
    spl=$(grep -m1 -oE "spl=[0-9]+" $f.txt | grep -oE "[0-9]+")
    res=$(grep RESULT $f.txt)
    [ -z "$res" ] && say "EMPTY LEG $f (runner law: assert the RESULT line)"
    say "r$R ctx=$CTX $NAME ${arm:-ARM-MISSING} s=${spl:-?} $res"
  done
 done
done
say "-- p48 retune datum: h256 16Q/2KV rows1 ctx8192 --"
for R in 1 2 3; do
  for LEG in p0 p1 cu96 f48; do
    f=q46logs/h256_${LEG}_r${R}
    case $LEG in
      p0) E="R3D_SPLIT_POLICY=0 R3D_FORCE_SPLITS=0";;
      p1) E="R3D_SPLIT_POLICY=1 R3D_FORCE_SPLITS=0";;
      cu96) E="R3D_SPLIT_POLICY=1 R3D_SPLIT_CU=96 R3D_FORCE_SPLITS=0";;
      f48) E="R3D_SPLIT_POLICY=1 R3D_FORCE_SPLITS=48";;
    esac
    env $E R3D_DEBUG_DECODE=1 BENCH_HD=256 ./q46bin/bench_attn_decode time 16 2 1 8192 15 8 > $f.txt 2>&1 || say "RUNFAIL $f"
    spl=$(grep -m1 -oE "spl=[0-9]+" $f.txt | grep -oE "[0-9]+")
    say "r$R h256 $LEG s=${spl:-?} $(grep RESULT $f.txt)"
  done
done
say "== ALL DONE $(date -u +%H:%M:%S) =="
