#!/usr/bin/env python3
# Q23 triage v2: per-symbol scratch placement, llvm-objdump amdgpu format:
#   symbol start:  ^<16hex> <name>:
#   instr:         \t<mnem> <ops>  // <16hex-pc>: <bytes> [<sym+off>]
#   branch target: operand is WORD-offset (imm*4 bytes) from NEXT insn, signed
# laws: case-insensitive hex; coverage assert refuses empty parses.
import re, sys

asm_path = sys.argv[1]
filt = sys.argv[2] if len(sys.argv) > 2 else ""

label_re  = re.compile(r"^([0-9a-fA-F]{4,16}) <(.+)>:")
ins_re    = re.compile(r"^\s+(\S+)(?:\s+(.*?))?\s*//\s*([0-9a-fA-F]{4,16}):")
hexre     = re.compile(r"0x([0-9a-fA-F]+)|^\s*(-?\d+)")
branch_re = re.compile(r"^(s_branch|s_cbranch\w*|v_cbranch\w+|s_setpc)$")
scratch_re = re.compile(r"^scratch_(load|store)")

syms = {}; order = []; cur = None; unparsed = 0
with open(asm_path) as f:
    for line in f:
        m = label_re.match(line)
        if m:
            cur = m.group(2); syms.setdefault(cur, []); 
            if cur not in order: order.append(cur)
            continue
        if cur is None: continue
        m = ins_re.search(line)
        if m:
            syms[cur].append((int(m.group(3), 16), m.group(1), m.group(2) or ""))
        elif line.strip() and not line.strip().startswith("//"):
            unparsed += 1

def decode_target(pc, mn, ops):
    if not ops.strip(): return None
    tok = ops.strip().split()
    try: v = int(tok[0], 10)
    except ValueError:
        h = hexre.search(ops); 
        if not h: return None
        v = int(h.group(1), 16) if h.group(1) else int(h.group(2))
    if mn.startswith("s_"):          # sopc branch: int16
        if v > 32767: v -= 65536
    else:                            # v_cbranch: int24
        if v > 8388607: v -= 16777216
    return pc + 4 + v * 4

def analyze(name, instrs):
    be = []; nb = 0; nut = 0
    for a, mn, ops in instrs:
        if not branch_re.match(mn): continue
        nb += 1
        t = decode_target(a, mn, ops)
        if t is None: nut += 1; continue
        if t <= a: be.append((t, a))
    scratch = [(a, mn) for a, mn, _ in instrs if scratch_re.match(mn)]
    inl = sum(1 for a, _ in scratch if any(t <= a <= b for t, b in be))
    return scratch, inl, nb, nut, be

matches = 0
for name in order:
    if filt and filt not in name: continue
    matches += 1
    instrs = syms[name]
    scratch, inl, nb, nut, be = analyze(name, instrs)
    if not scratch:
        continue
    stores = sum(1 for _, mn in scratch if mn.startswith("scratch_store"))
    idx = {a: i for i, (a, _, _,) in enumerate([(a, mn, o) for a, mn, o in instrs])}
    def near(a, pat, w=25):
        i = idx[a]
        lo, hi = max(0, i - w), min(len(instrs), i + w)
        return any(re.search(pat, instrs[j][1]) or re.search(pat, instrs[j][2]) for j in range(lo, hi))
    glb  = sum(1 for a, _ in scratch if near(a, r"global_(load|store)"))
    dwr  = sum(1 for a, _ in scratch if near(a, r"ds_write"))
    wmma = sum(1 for a, _ in scratch if near(a, r"v_wmma"))
    expp = sum(1 for a, _ in scratch if near(a, r"v_exp"))
    print(f"{name[:86]}")
    print(f"  scratch {len(scratch)} (st {stores}/ld {len(scratch)-stores}) "
          f"IN-LOOP {inl} | branches {nb} unparsed-targets {nut} backedges {len(be)}")
    print(f"  ctx: global-traffic {glb}  ds_write {dwr}  wmma {wmma}  exp2 {expp}")

tot = sum(len(syms[n]) for n in order)
assert tot > 1000, f"parse coverage collapse: only {tot} instrs parsed"
print(f"COVERAGE: {len(order)} symbols, {tot} instrs, {unparsed} unparsed lines, {matches} matched filter")
