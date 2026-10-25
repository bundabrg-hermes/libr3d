#!/usr/bin/env python3
"""Q18b A/B adjudicator: reads the AB lines of q18_gates.log, groups by shape+arm,
prints medians, per-arm spread (min..max) and the delta vs the shape's baseline arm,
and states whether the spreads are DISJOINT (the ledger's adjudication standard)."""
import re, sys, statistics as st
from collections import defaultdict

pat = re.compile(r"AB r=(\d+) arm=(\S+) ROW M=(\d+) K=(\d+) N=(\d+) rot=(\d+) reps=\d+ "
                 r"\| t_med=([\d.]+) ms\s+(\S+) ([\d.]+) GB/s\s+nonzero=(\d+)/(\d+)")
data = defaultdict(dict)     # (shape) -> arm -> [t_med...]
live = defaultdict(lambda: defaultdict(set))   # (shape) -> arm -> set of nonzero flags
for line in open(sys.argv[1]):
    m = pat.search(line)
    if not m: continue
    r, arm, M, K, N, rot, tms, _, gbs, nz, tot = m.groups()
    shape = "M=%s K=%s N=%s rot=%s" % (M, K, N, rot)
    data[shape].setdefault(arm, []).append(float(tms))
    live[shape][arm].add(int(nz) > 0)

for shape, arms in data.items():
    print("\n=== %s ===" % shape)
    if any(not all(v) for v in live[shape].values()):
        print("  !! DEAD-ROW ARM PRESENT -- rate is a lie for that arm")
    cands = [a for a in ("base", "def") if a in arms]
    if not cands:
        print("  (no baseline arm in this group)")
        continue
    for bname in cands:
        b = arms[bname]
        bmed = st.median(b)
        print("  %-6s med=%7.3f ms  spread %.3f..%.3f  n=%d" % (bname, bmed, min(b), max(b), len(b)))
        for arm, v in sorted(arms.items()):
            if arm == bname: continue
            med = st.median(v)
            d = (bmed - med) / bmed * 100.0
            disjoint = min(v) > max(b) or max(v) < min(b)
            verdict = ("WIN " if d > 0 else "LOSS") if disjoint else "overlap"
            print("  %-6s med=%7.3f ms  spread %.3f..%.3f  n=%d  %+6.1f%%  %s"
                  % (arm, med, min(v), max(v), len(v), d,
                     ("%s (spreads disjoint)" % verdict) if disjoint else verdict))
