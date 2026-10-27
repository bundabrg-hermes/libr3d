#!/usr/bin/env bash
# q44_serve.sh -- serve A/B: v159 lever-0 vs lever-1, same binary, container env apart.
# SERIAL runner, distinct legs, crossover order per round (l0,l1,l1,l0 alternation by
# round cancels drift). Readiness = /tokenize 200 (never /health). Throwaway round per
# boot (settle), then timed probes. Liveness per round from docker logs (stderr law).
set -uo pipefail
S=~/scratch/r3d/q44
LOG=$S/serve.log; : > $LOG
U=http://localhost:9008
say() { echo "$@" | tee -a $LOG; }
wait_ready() { for i in $(seq 1 120); do
    code=$(curl -s -m 3 $U/tokenize -H "Content-Type: application/json" -d "{\"prompt\":\"hi\"}" -o /dev/null -w "%{http_code}" 2>/dev/null)
    [ "$code" = "200" ] && return 0; sleep 4; done; return 1; }
boot() { # boot <lmap>
  local lm=$1
  docker stop spec9g > /dev/null 2>&1; docker rename spec9g spec9g_q44_prev 2>/dev/null
  docker rm spec9g_q44_prev > /dev/null 2>&1
  python3 $S/q44_serve_body.py $S/body_l$lm.json $lm >> $LOG
  local id; id=$(curl -s --unix-socket /var/run/docker.sock -X POST -H "Content-Type: application/json" \
      --data-binary @$S/body_l$lm.json "http://localhost/containers/create?name=spec9g" | python3 -c "import sys,json;print(json.load(sys.stdin).get('Id',''))")
  [ -n "$id" ] || { say "!! create failed lm=$lm"; return 1; }
  curl -s --unix-socket /var/run/docker.sock -X POST -H "Content-Type: application/json" -d "{}" \
      http://localhost/containers/spec9g/start >> $LOG
  docker inspect spec9g --format '{{.State.Status}} {{json .Config.Env}}' | grep -o "R3D_Q8X_LMAP=[01]" >> $LOG
  wait_ready || { say "!! not ready lm=$lm -- ABORT (v153 restore manual)"; return 1; }
  docker exec spec9g md5sum /rhome/kernels/libr3d.so >> $LOG 2>&1
  say "booted lm=$lm $(date -u +%H:%M:%S)"
}
probe_round() { # probe_round <lmap> <round>
  local lm=$1 r=$2
  python3 ~/projects/libr3d/libr3d/tools/decode_probe.py --url $U --ctx 3072 --reps 3 --tag lm${lm}-r${r} >> $LOG 2>&1 \
      && say "dec lm$lm r$r rc=0" || say "dec lm$lm r$r FAIL"
  docker logs --tail 40 spec9g 2>&1 | grep "arm launched" | sort -u | sed "s/^/liveness lm$lm r$r: /" >> $LOG
}
say "== serve A/B start $(date -u +%H:%M:%S) =="
for r in 1 2 3; do
  # order: r1 l0->l1, r2 l1->l0, r3 l0->l1 (crossover by round cancels drift)
  if [ $((r % 2)) -eq 1 ]; then order="0 1"; else order="1 0"; fi
  for lm in $order; do
    boot $lm || exit 1
    python3 ~/projects/libr3d/libr3d/tools/decode_probe.py --url $U --ctx 3072 --reps 1 --tag settle >/dev/null 2>&1  # throwaway settle
    probe_round $lm $r
  done
done
say "== SERVE AB DONE $(date -u +%H:%M:%S) =="
