#!/usr/bin/env python3
"""q44_serve_body.py <out.json> <lmap> -- rebuild the spec9g create-body from the
LIVE inspect snapshot (stale-template restart-loop law), swap the libr3d mount to
build_v159, and pin R3D_Q8X_LMAP + R3D_DEBUG_GEMM as the only deltas."""
import json, sys
src = json.load(open("/home/brendan/scratch/r3d/q44_spec9g_inspect_live.json"))[0]
out_path, lmap = sys.argv[1], sys.argv[2]
c, h = src["Config"], src["HostConfig"]
env = [e for e in c["Env"] if not e.startswith(("R3D_Q8X_LMAP", "R3D_DEBUG_GEMM"))]
if lmap == "close":
    binds = list(h["Binds"])          # verbatim v153 restore, stock env
else:
    env += [f"R3D_Q8X_LMAP={lmap}", "R3D_DEBUG_GEMM=1"]
    binds = [b.replace("build_v153/radiance_home", "build_v159/radiance_home") for b in h["Binds"]]
    assert any("build_v159" in b for b in binds), "bind swap failed"
body = {
    "Image": c["Image"],
    "Cmd": c["Cmd"],
    "ExposedPorts": c.get("ExposedPorts"),
    "WorkingDir": c.get("WorkingDir"),
    "Env": env,
    "HostConfig": {
        "Binds": binds,
        "Devices": h["Devices"],
        "PortBindings": h["PortBindings"],
        "NetworkMode": h.get("NetworkMode", "default"),
        "IpcMode": h.get("IpcMode", "private"),
        "ShmSize": h.get("ShmSize", 67108864),
        "RestartPolicy": h.get("RestartPolicy", {"Name": "unless-stopped"}),
        "Memory": h.get("Memory", 0),
    },
}
json.dump(body, open(out_path, "w"), indent=1)
print(f"wrote {out_path} (lmap={lmap})")
