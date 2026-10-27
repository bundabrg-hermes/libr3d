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
for name, armenv in [("q46_vfy0", ["R3D_SPLIT_POLICY=0"]),
                     ("q46_vfy1", ["R3D_SPLIT_POLICY=1", "R3D_SPLIT_CU=96"])]:
    binds = [b.replace("build_v153/radiance_home", "build_v155test/radiance_home")
             for b in src["HostConfig"]["Binds"]]
    env = [e for e in src["Config"]["Env"]
           if not e.startswith(("R3D_SPLIT_POLICY", "R3D_DEBUG_DECODE", "R3D_SPLIT_CU"))]
    env += armenv + ["R3D_DEBUG_DECODE=1"]
    body = {"Image": src["Config"]["Image"], "Env": env, "Cmd": src["Config"]["Cmd"],
            "Entrypoint": src["Config"].get("Entrypoint"),
            "ExposedPorts": src["Config"].get("ExposedPorts"),
            "HostConfig": {"Binds": binds,
                           "Devices": src["HostConfig"].get("Devices", []),
                           "PortBindings": {"9008/tcp": [{"HostPort": "9009"}]},
                           "ShmSize": src["HostConfig"].get("ShmSize", 67108864)}}
    sh("docker", "rm", "-f", name)
    r = api("POST", f"/containers/create?name={name}", body)
    assert "Id" in r, (name, r)
    api("POST", f"/containers/{name}/start", {})
    ok = False
    for _ in range(60):
        try:
            req = urllib.request.Request("http://localhost:9009/tokenize",
                                         data=b'{"prompt":"hi"}',
                                         headers={"Content-Type": "application/json"})
            if urllib.request.urlopen(req, timeout=3).status == 200:
                ok = True
                break
        except Exception:
            pass
        time.sleep(4)
    p = {"prompt": uuid.uuid4().hex[:8] + "\nThe lighthouse keeper logged the weather.",
         "n_predict": 256, "temperature": 0, "seed": 7, "stream": False}
    req = urllib.request.Request("http://localhost:9009/completion",
                                 data=json.dumps(p).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        r.read()
    time.sleep(2)
    logs = sh("docker", "logs", name).stdout + sh("docker", "logs", name).stderr
    pl = [l for l in logs.splitlines() if "split-policy" in l]
    sp = sorted(set(l for l in logs.splitlines() if "splitk" in l))
    print(name, "ready:", ok)
    print("  policy:", pl[:2])
    print("  distinct splitk lines:", len(sp), sp[:2])
    sh("docker", "rm", "-f", name)
print("VERIFY_DONE")
