#!/usr/bin/env python3
"""q44_walk_audit.py <lm.asm> — walk-region + saveexec audit on the integrated
lane-map kernel's disassembly (addresses from trailing comments, window-18 law)."""
import re, sys
lines = open(sys.argv[1]).read().splitlines()
fn = re.match(r"^0*([0-9a-f]+) <(\S+)>:", lines[0])
base, sym = int(fn.group(1), 16), fn.group(2)
ins = []
for l in lines[1:]:
    if re.match(r"^0*[0-9a-f]{6,} <\S+>:", l): break
    m = re.search(r"// ([0-9A-Fa-f]{8,14}): ", l)
    if m: ins.append((int(m.group(1), 16) - base, l))
edges = []
for off, l in ins:
    m = re.search(r"s_(?:branch|cbranch\w*) \d+\s+// [0-9A-Fa-f]{8,14}: \S+ <" + sym + r"\+0x([0-9A-Fa-f]+)>", l)
    if m:
        tgt = int(m.group(1), 16)
        if tgt < off: edges.append((tgt, off))
if not edges: sys.exit("no back edges?! coverage guard")
lo, hi = max(edges, key=lambda e: e[1] - e[0])
xop = re.compile(r"ds_bpermute|v_mov_b32_dpp|permlanex16")
n_in = sum(1 for o, l in ins if lo <= o <= hi and xop.search(l))
def is_restore(l):
    return bool(re.search(r"s_or_b(32|64)\s+exec(_lo)?,\s*exec(_lo)?,\s*s\d+", l)
                or re.search(r"s_mov_b32\s+exec(_lo)?,\s*s\d+", l))
depth = 0; xop_under = 0; regions = []; kinds = {}; stack = []
for o, l in ins:
    m = re.search(r"s_and_(?:not1_)?saveexec_b(32|64)\s+(s\d+)", l)
    if m:
        depth += 1; regions.append(o); stack.append(m.group(2))
    if depth and xop.search(l): xop_under += 1
    if depth:
        m = re.search(r"// [0-9A-Fa-f]{8,14}: \S+\s+([a-z_0-9.]+)", l)
        if m: kinds[m.group(1)] = kinds.get(m.group(1), 0) + 1
    if depth and is_restore(l) and stack:
        # restore names the saved register of the innermost region
        if stack[-1] in l: stack.pop(); depth -= 1
tot = sum(1 for o, l in ins if xop.search(l))
print(f"kernel: {sym[:70]}")
print(f"walk loop (widest back edge): 0x{lo:x}..0x{hi:x}")
print(f"total XOP: {tot} | XOP in walk loop: {n_in} | XOP under saveexec: {xop_under} | regions: {len(regions)} depth_open_at_end: {depth}")
if kinds:
    top = dict(sorted(kinds.items(), key=lambda x: -x[1])[:10])
    print("mnemonics under saveexec:", top)
print("VERDICT:", "CLEAN (zero cross-lane in walk; zero under predication)"
      if n_in == 0 and xop_under == 0 else "FINDING")
