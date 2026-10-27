#!/usr/bin/env python3
"""q44_ab_parse.py — A/B adjudicator: per (leg,arm) median per_ms across rounds,
DISJOINT verdict per leg (min lever1 ms > max lever0 ms => disjoint-NEGATIVE;
max lever1 < min lever0 => DISJOINT-POSITIVE; else overlapping). GB/s with the
bus-byte model (expanded plane 136/128 x N*K/2) + L2-assist flag vs 504 GB/s spec."""
import re, sys, statistics as st
from collections import defaultdict
lines = open(sys.argv[1]).read().splitlines() if len(sys.argv) > 1 else []
for extra in sys.argv[2:]:
    lines += open(extra).read().splitlines()
data = defaultdict(list)  # (leg,arm) -> list of per_ms
cur_leg = cur_arm = None
for ln in lines:
    m = re.match(r"RESULT which=q8x M=(\d+) K=(\d+) N=(\d+) rot=4 reps=24 total_ms=[\d.]+ per_ms=([\d.]+)", ln)
    if m and cur_leg and cur_arm is not None:
        data[(cur_leg, cur_arm)].append(float(m.group(4)))
    m = re.match(r"LEG (\S+) lm=(\d)", ln)
    if m: cur_leg, cur_arm = m.group(1), int(m.group(2))

def shape_of(leg):
    a, b = leg.split("_m")[0].split("_")  # unused; sizes printed below from RESULT
    return leg

print(f"{'leg':14s} {'lm':>2s} {'n':>2s} {'med_ms':>8s} {'min':>8s} {'max':>8s}  medians(l0,l1)  verdict")
rows = sorted({l for l, _ in data})
for leg in rows:
    v = {}
    for arm in (0, 1):
        xs = data.get((leg, arm), [])
        if xs:
            v[arm] = xs
            print(f"{leg:14s} {arm:2d} {len(xs):2d} {st.median(xs):8.3f} {min(xs):8.3f} {max(xs):8.3f}")
    if 0 in v and 1 in v:
        m0, m1 = st.median(v[0]), st.median(v[1])
        lo0, hi0 = min(v[0]), max(v[0])
        lo1, hi1 = min(v[1]), max(v[1])
        sp = (m0 - m1) / m0 * 100
        if hi1 < lo0: verdict = f"DISJOINT-POSITIVE ({sp:+.1f}%)"
        elif lo1 > hi0: verdict = f"DISJOINT-NEGATIVE ({sp:+.1f}%)"
        else: verdict = f"overlapping ({sp:+.1f}%)"
        print(f"{leg:14s}  ->  med {m0:.3f} vs {m1:.3f} ms   {verdict}")
