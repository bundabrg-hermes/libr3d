import json, subprocess, time, urllib.request, uuid
SOCK = "/var/run/docker.sock"

def api(method, path, body=None):
    cmd = ["curl", "-s", "--unix-socket", SOCK, "-X", method,
           "-H", "Content-Type: application/json", f"http://localhost{path}"]
    if body is not None:
        cmd += ["-d", json.dumps(body)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except Exception:
        return {"raw": r.stdout[:300]}

sh = lambda *a: subprocess.run(a, capture_output=True, text=True)
src = api("GET", "/containers/spec9g/json")
assert "spec9g " not in sh("docker", "ps", "--format", "{{.Names}}\n").stdout or True
binds = [b.replace("build_v153/radiance_home", "build_v155test/radiance_home")
         for b in src["HostConfig"]["Binds"]]
env = [e for e in src["Config"]["Env"]
       if not e.startswith(("R3D_SPLIT_POLICY", "R3D_DEBUG_DECODE"))]
env += ["R3D_SPLIT_POLICY=1", "R3D_DEBUG_DECODE=1"]
cmd = list(src["Config"]["Cmd"])
i = cmd.index("--num-speculative-tokens")
cmd[i + 1] = "0"
body = {"Image": src["Config"]["Image"], "Env": env, "Cmd": cmd,
        "Entrypoint": src["Config"].get("Entrypoint"),
        "ExposedPorts": src["Config"].get("ExposedPorts"),
        "HostConfig": {"Binds": binds,
                       "Devices": src["HostConfig"].get("Devices", []),
                       "PortBindings": src["HostConfig"].get("PortBindings", {}),
                       "ShmSize": src["HostConfig"].get("ShmSize", 67108864)}}
sh("docker", "rm", "-f", "q46_spec0")
r = api("POST", "/containers/create?name=q46_spec0", body)
assert "Id" in r, r
api("POST", "/containers/q46_spec0/start", {})
ok = False
for _ in range(60):
    try:
        req = urllib.request.Request("http://localhost:9008/tokenize",
                                     data=b'{"prompt":"hi"}',
                                     headers={"Content-Type": "application/json"})
        if urllib.request.urlopen(req, timeout=3).status == 200:
            ok = True
            break
    except Exception:
        pass
    time.sleep(4)
print("ready:", ok, flush=True)
for i, n in enumerate((128, 512)):
    p = {"prompt": uuid.uuid4().hex[:8] + "\nThe lighthouse keeper logged the weather.",
         "n_predict": n, "temperature": 0, "seed": 7, "stream": False}
    req = urllib.request.Request("http://localhost:9008/completion",
                                 data=json.dumps(p).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        r.read()
    time.sleep(2)
logs = sh("docker", "logs", "q46_spec0").stdout + sh("docker", "logs", "q46_spec0").stderr
sl = [l for l in logs.splitlines() if "splitk" in l]
pl = [l for l in logs.splitlines() if "split-policy" in l]
dl = [l for l in logs.splitlines() if " done " in l and "gen=" in l]
print("splitk lines:", len(sl))
print("policy lines:", pl[:3])
print("sample splitk:", sl[:2])
print("done:", dl)
sh("docker", "rm", "-f", "q46_spec0")
print("PROBE_DONE")
