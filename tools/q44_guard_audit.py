#!/usr/bin/env python3
"""q44_guard_audit.py <asm> — linear EXEC-guard replay: any cross-lane op executing
between a saveexec-mask and its restore = poison (window-46 law). b32 exec_lo form."""
import re, sys
lines = open(sys.argv[1]).read().splitlines()
fn = re.match(r"^0*([0-9a-f]+) <(\S+)>:", lines[0])
base, sym = int(fn.group(1), 16), fn.group(2)
xop = re.compile(r"ds_bpermute|v_mov_b32_dpp|permlanex16")
guarded = 0; seq = []
cur = 0
tot_x = 0
open_addr = None
for l in lines[1:]:
    if re.match(r"^[0-9a-f]{7,} <\S+>:", l): break
    if re.search(r"s_and(_not1)?_saveexec_b32\s+s\d+,\s*s\d+", l):
        if cur: seq.append(("NEST", l))
        cur = 1
    elif re.search(r"s_or_b32\s+exec_lo,\s*exec_lo,\s*s\d+", l) or re.search(r"s_mov_b32\s+exec_lo,\s*s\d+", l):
        cur = 0
    elif xop.search(l):
        tot_x += 1
        if cur:
            guarded += 1
            seq.append(("XOP@GUARD", l[:80]))
print(f"{sym[:60]}: XOP={tot_x} XOP-under-exec-guard={guarded} nests={sum(1 for k,_ in seq if k=='NEST')}")
for k, s in seq[:4]: print("  ", k, s.strip()[:100])
print("VERDICT:", "CLEAN" if guarded == 0 and not any(k == 'NEST' for k, _ in seq) else "FINDING")
