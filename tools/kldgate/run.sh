#!/usr/bin/env bash
# kldgate run -- serve A/B prefill-logit drift gate.
#
# Arms (both the same container, same weights, same card, one at a time):
#   base : R3D_NO_WMMA=1  (the oracle-green scalar prefill tile)
#   test : stock          (the WMMA m64 prefill tile, what ships)
#
# For future kernel work the arms generalize: base = the tip that shipped,
# test = the working tree's build. Run from anywhere; assumes rdna3-dev, assumes the
# serve recipe below matches the deployment (it was read from
# `docker inspect libr3d-serve` -- re-check if that container's birth command
# ever changes).
set -euo pipefail

IMG=radiance-dev:v1.8.1
RHOME=/home/devuser/projects/libr3d/build/radiance_home
NAME=libr3d-serve
OUT=/tmp/kldgate-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"

stop_serve() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  # a GPU container's removal can lag its kill (device teardown); wait until
  # the name is actually free, or the next docker run conflicts
  for i in $(seq 1 30); do
    docker inspect "$NAME" >/dev/null 2>&1 || break
    sleep 2
  done
}

start_serve() {
  local extra_env="$1"
  local eargs=()
  [ -n "$extra_env" ] && eargs=(-e "$extra_env")
  docker run -d --name "$NAME" --shm-size 4g \
    -e ROCR_VISIBLE_DEVICES=0 "${eargs[@]}" \
    -v /srv/models/rad:/models \
    -v "$RHOME:/rhome" \
    -p 9005:9005 \
    --device /dev/kfd --device /dev/dri \
    "$IMG" /opt/radiance/bin/radiance \
      --model /models/minicpm5-2b-bf16.rad \
      --radiance-home /rhome:/opt/radiance/share/radiance \
      --num-speculative-tokens 0 --max-model-len 16384 \
      --host 0.0.0.0 --port 9005 --live
  echo "waiting for $NAME ($extra_env) ..."
  for i in $(seq 1 120); do
    if curl -sf http://localhost:9005/v1/models >/dev/null; then echo "up after ${i} polls"; return 0; fi
    sleep 2
  done
  echo "serve did not come up; last logs:"; docker logs --tail 30 "$NAME"; exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# arm A: the base (scalar tile)
stop_serve
start_serve "R3D_NO_WMMA=1"
python3 "$SCRIPT_DIR/kldgate.py" collect --arm scalar-base --out "$OUT/base.jsonl"

# arm B: the shipped default (WMMA tile)
stop_serve
start_serve ""
python3 "$SCRIPT_DIR/kldgate.py" collect --arm wmma --out "$OUT/test.jsonl"
python3 "$SCRIPT_DIR/kldgate.py" score --trajectory "$OUT/base.jsonl" --out "$OUT/forced.jsonl"

# compare (gate verdict is the exit code)
set +e
python3 "$SCRIPT_DIR/kldgate.py" compare --base "$OUT/base.jsonl" \
  --free "$OUT/test.jsonl" --forced "$OUT/forced.jsonl" \
  --report "$OUT/report.md" | tee "$OUT/verdict.txt"
rc=$?
set -e
echo "gate exit $rc; artifacts in $OUT"
exit $rc
