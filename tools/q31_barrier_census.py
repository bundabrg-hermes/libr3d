#!/usr/bin/env python3
"""Q31 census: shipped-code audit of the WGP global-store/barrier law.

Law (RDNA3 WGP, iserver-sglang triage P-B): a WG spans two L0 caches; GLOBAL
stores are vscnt-counted and `s_waitcnt lgkmcnt(0); s_barrier` does NOT wait
them. Exposure class = (global store) -> s_barrier -> (global load of the same
data by another thread) with no intervening `s_waitcnt_vscnt null,0x0`.
Scratch stores are thread-private and lgkmcnt-counted: they are NOT in the
class (first-pass censuses that count scratch_store report only spill storms).

Pipeline: carve embedded gfx1100 code objects out of the shipped libr3d.so
(.hip_fatbin section; 3 ISA targets x per-TU objects), disassemble with
llvm-objdump, then per-function linear scan: pending global stores since the
last full vscnt drain, flagged at every s_barrier. Linear scan is conservative
across CFG merges -- a flagged site must be read in context; store location
relative to the last barrier distinguishes in-loop publishes from kernel-exit
stores (kernel exit drains vscnt implicitly; stream order covers the rest).

Usage: python3 q31_barrier_census.py /path/to/libr3d.so [--gfx gfx1100]
Requires: llvm-objdump, llvm-readelf, c++filt on PATH.
"""
import os, re, sys, struct, subprocess, tempfile

# Window-52 drift repair: R3D_LLVM_BIN=<dir> overrides the llvm tool dir;
# default is the system PATH (previously an implicit ROCm-10.1 pin).
LLVM = os.environ.get("R3D_LLVM_BIN", "").strip()
def _tool(nm): return (LLVM + "/" + nm) if LLVM else nm

def carve(path, outdir, gfx):
    data = open(path, "rb").read()
    sec = subprocess.run(["objcopy", "-O", "binary", "--only-section=.hip_fatbin",
                          path, "/dev/stdout"], capture_output=True).stdout or data
    offs, i = [], 0
    while True:
        i = sec.find(b"\x7fELF", i)
        if i < 0:
            break
        if i % 8 == 0 and len(sec) > i + 64:
            if struct.unpack_from("<H", sec, i + 16)[0] == 3:
                shoff = struct.unpack_from("<Q", sec, i + 32)[0]
                if shoff < len(sec) - i:
                    offs.append(i)
        i += 4
    names = {}
    for j, off in enumerate(offs):
        end = offs[j + 1] if j + 1 < len(offs) else len(sec)
        p = os.path.join(outdir, f"co_{j:02d}.bin")
        open(p, "wb").write(sec[off:end])
        n = subprocess.run([_tool("llvm-readelf"), "--notes", p], capture_output=True, text=True).stdout
        # Match BOTH target-note forms: ROCm-era ".target:" and current "amdhsa.target:".
        tgt = re.search(r"(?:amdhsa\.target|\.target):\s+(\S+)", n)
        if tgt and gfx in tgt.group(1):
            names[p] = n
    return names

STORE = re.compile(r"\b(?:buffer_store|global_store|flat_store)")  # NOT scratch_store
HDR = re.compile(r"^[0-9a-fA-F]+ <(.+)>:")  # window-52: UPPER-CASE hex from current llvm

def census_asm(path):
    """Yield (func, barriers, [(line, pending_stores), ...]) per kernel function."""
    cur, barriers, pending, sites = None, 0, 0, []
    def flush():
        if cur and "virtual" not in cur:
            yield_row.append((cur, barriers, sites))
    yield_row = []
    for num, line in enumerate(open(path), 1):
        m = HDR.match(line)
        if m:
            flush()
            cur, barriers, pending, sites = m.group(1), 0, 0, []
            continue
        if re.search(r"s_waitcnt_vscnt\s+null,\s*0x0", line):
            pending = 0
            continue
        if STORE.search(line):
            pending += 1
        elif re.search(r"\bs_barrier\b", line):
            barriers += 1
            if pending > 0:
                sites.append((num, pending))
    flush()
    return yield_row

def main():
    so = sys.argv[1]
    gfx = sys.argv[sys.argv.index("--gfx") + 1] if "--gfx" in sys.argv else "gfx1100"
    with tempfile.TemporaryDirectory() as td:
        objs = carve(so, td, gfx)
        print(f"{so}: {gfx} code objects: {len(objs)}")
        tot_f = tot_b = 0
        exposed = 0
        for p in sorted(objs):
            s = subprocess.run([_tool("llvm-objdump"), "-d", "--mcpu=" + gfx, p],
                               capture_output=True, text=True).stdout
            fp = os.path.join(td, os.path.basename(p) + ".s")
            open(fp, "w").write(s)
            for fn, b, sites in census_asm(fp):
                if b:
                    tot_f += 1
                    tot_b += b
                    if sites:
                        exposed += 1
                        print(f"REVIEW {fn[:92]}  {len(sites)}/{b} barriers, sites {sites}")
        print(f"functions with s_barrier: {tot_f}; barriers: {tot_b}; with pending global stores: {exposed}")
        print("NOTE: pending>0 at a barrier is conservative (linear scan); read each site in context.")

if __name__ == "__main__":
    main()
