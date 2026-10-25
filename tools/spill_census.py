#!/usr/bin/env python3
"""Spill census: per-kernel VGPR spill / private-segment counts for the gfx1100 code
objects in a libr3d build. Run inside radiance-dev (needs llvm-objcopy + llvm-readelf on
PATH, e.g. /opt/rocm/llvm/bin):

    docker run --rm -v $PWD:/work -w /work radiance-dev \
      bash -c 'export PATH=/opt/rocm/llvm/bin:$PATH && python3 tools/spill_census.py [build]'

Prints kernels with spills or a private segment, worst first. A spill cliff is invisible
to a timing table that only times the shapes its fixture holds; census the whole tree.
"""
import glob
import os
import re
import struct
import subprocess
import sys
import tempfile

def main():
    build = sys.argv[1] if len(sys.argv) > 1 else "build"
    src = os.path.join(build, "CMakeFiles", "libr3d.dir", "src")
    if not os.path.isdir(src):
        sys.exit(f"no objects under {src} (build the library first)")
    want = os.environ.get("CENSUS_ARCH", "gfx1100")
    rows = []
    for o in sorted(glob.glob(os.path.join(src, "*.o"))):
        fb = tempfile.mktemp(suffix=".fb")
        r = subprocess.run(["llvm-objcopy", "--dump-section", ".hip_fatbin=" + fb, o,
                            "/tmp/spill_census_nul.o"], capture_output=True, text=True)
        if r.returncode or not os.path.exists(fb):
            continue
        data = open(fb, "rb").read()
        i = 0
        while True:
            i = data.find(b"\x7fELF", i + 1)
            if i < 0 or len(data) < i + 64:
                break
            if data[i + 4] != 2 or struct.unpack_from("<H", data, i + 18)[0] != 224:
                continue
            co = data[i:]
            p = tempfile.mktemp(suffix=".elf")
            open(p, "wb").write(co)
            out = subprocess.run(["llvm-readelf", "--notes", p],
                                 capture_output=True, text=True).stdout
            os.unlink(p)
            if "amdhsa.kernels" not in out:
                i += 8
                continue
            t = re.search(r"amdhsa\.target:\s+(\S+)", out)
            arch = t.group(1).rsplit("-", 1)[-1] if t else "?"
            for block in out.split("\n  - .args:")[1:]:
                kn = re.search(r"\.name:\s+(\S+)", block)
                sp = re.search(r"\.vgpr_spill_count:\s+(\d+)", block)
                pr = re.search(r"\.private_segment_fixed_size:\s+(\d+)", block)
                if kn:
                    rows.append((os.path.basename(o), kn.group(1), arch,
                                 int(sp.group(1)) if sp else 0,
                                 int(pr.group(1)) if pr else 0))
            i += 8
        os.unlink(fb)
    seen = {(r[0], r[1]): r for r in rows if r[2] == want}
    g = list(seen.values())
    bad = [r for r in g if r[3] > 0 or r[4] > 0]
    print(f"{want} kernels: {len(g)}  spilling/private: {len(bad)}")
    for r in sorted(bad, key=lambda r: -(r[3] + r[4])):
        print(f"{r[3]:>6} spill {r[4]:>6} priv  {r[0]:<30} {r[1]}")

if __name__ == "__main__":
    main()
