#!/usr/bin/env python3
"""Q46 serve decode A/B: build_v154 mounted, arms = R3D_SPLIT_POLICY 0|1 ONLY.

Recipe (w43/w36/w34 laws): two pre-created containers from spec9g's own live spec
(same image, same args), Binds -> build_v154/radiance_home, Env += the arm +
R3D_DEBUG_DECODE=1. Only one runs at a time (port 9008). Per switch: /tokenize-200
readiness, one throwaway gen=512 round, settle until a throwaway rate is sane
(graph-record warmup reads 9-21 tok/s -- that is warmup, not the arm), then the
measured round: fresh uuid salt, gen=512, tok/s parsed from the container's own
done line (docker logs -t, request id from the response body; `docker logs
--since` is broken on cat). Order: A1 B1 A2 B2 A3 B3 (strict alternation).
NEVER touches spec9g (stopped by caller) or the rollback container; cleanup removes
the two q46 containers.
"""
import json, os, re, socket, subprocess, sys, time, urllib.request, uuid

SOCK = "/var/run/docker.sock"


def api(method, path, body=None):
    """docker daemon over unix socket via curl (raw-socket keep-alive hangs)."""
    cmd = ["curl", "-s", "--unix-socket", SOCK, "-X", method,
           "-H", "Content-Type: application/json", f"http://localhost{path}"]
    if body is not None:
        cmd += ["-d", json.dumps(body)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except Exception:
        return {"raw": r.stdout[:400]}


def sh(*args):
    return subprocess.run(args, capture_output=True, text=True)


def ensure_arm_container(name, arm_env):
    """Create if absent (DO NOT start: both arms want port 9008; the loop starts
    the chosen arm only after stopping the other)."""
    r = api("GET", f"/containers/{name}/json")
    if "Id" in r and "message" not in r:
        return
    src = api("GET", "/containers/spec9g/json")
    binds = [b.replace("build_v153/radiance_home", "build_v154/radiance_home")
             for b in src["HostConfig"].get("Binds", [])]
    env = [e for e in src["Config"].get("Env", [])
           if not e.startswith(("R3D_SPLIT_POLICY", "R3D_DEBUG_DECODE"))]
    env += list(arm_env) + ["R3D_DEBUG_DECODE=1"]
    hc = {"Binds": binds,
          "Devices": src["HostConfig"].get("Devices", []),
          "PortBindings": src["HostConfig"].get("PortBindings", {}),
          "ShmSize": src["HostConfig"].get("ShmSize", 67108864)}
    body = {"Image": src["Config"]["Image"], "Env": env, "Cmd": src["Config"]["Cmd"],
            "Entrypoint": src["Config"].get("Entrypoint"),
            "ExposedPorts": src["Config"].get("ExposedPorts"), "HostConfig": hc}
    r = api("POST", f"/containers/create?name={name}", body)
    if "Id" not in r:
        print(f"create {name} FAILED: {r}", flush=True)
        sys.exit(1)


def start_arm(name):
    r = api("GET", f"/containers/{name}/json")
    if r.get("State", {}).get("Status") == "running":
        return True
    api("POST", f"/containers/{name}/start", {})
    r = api("GET", f"/containers/{name}/json")
    ok = r.get("State", {}).get("Status") == "running"
    if not ok:
        print(f"  start {name} FAILED: {r.get('State')}", flush=True)
    return ok


def ready():
    last = None
    for i in range(60):
        for body in ({"prompt": "hi"}, {"content": "hi"}):
            try:
                req = urllib.request.Request(
                    "http://localhost:9008/tokenize",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=3) as r:
                    if r.status == 200:
                        print(f"  ready after {i+1} probes ({list(body)[0]}-body)", flush=True)
                        return True
            except Exception as e:
                if str(e) != last and i % 10 == 0:
                    last = str(e)
                    print(f"  probe#{i}: {e}", flush=True)
        time.sleep(4)
    return False


def decode_round(name):
    salt = uuid.uuid4().hex[:12]
    payload = {"prompt": salt + "\nThe lighthouse keeper logged the weather.",
               "n_predict": 512, "temperature": 0, "seed": 7, "stream": False}
    req = urllib.request.Request(
        "http://localhost:9008/completion", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    before = sh("docker", "logs", name).stdout + sh("docker", "logs", name).stderr
    with urllib.request.urlopen(req, timeout=600) as r:
        r.read()
    after = sh("docker", "logs", name).stdout + sh("docker", "logs", name).stderr
    # the completion response carries no id: tie by NEW done-lines (gen=512)
    new = [l for l in after.splitlines()
           if l not in set(before.splitlines()) and " done " in l and "gen=512" in l]
    for line in reversed(new):
        m = re.search(r"([0-9.]+)\s+tok/s", line)
        if m:
            return float(m.group(1)), line
    return None, f"no new gen=512 done line (new lines: {len(new)})"


def settle(name):
    """throwaway rounds until rate is sane (graph-record passes done), >=2 sane."""
    sane = 0
    for i in range(10):
        r, _ = decode_round(name)
        print(f"  settle {name} #{i}: {r}", flush=True)
        if r is not None and r >= 30:
            sane += 1
            if sane >= 2:
                return True
        else:
            time.sleep(30)
    return False


def liveness(name, pol_env):
    logs = sh("docker", "logs", name).stdout + sh("docker", "logs", name).stderr
    arm = [l for l in logs.splitlines() if "split-policy arm=" in l]
    spls = [l for l in logs.splitlines() if "DBG attn-decode splitk" in l]
    want = "arm=wave" if "POLICY=1" in pol_env else "arm=fixed"
    ok = bool(arm) and want in arm[-1]
    print(f"  liveness {name}: {arm[-1] if arm else 'ARM LINE MISSING'} "
          f"({len(spls)} splitk launch DBG lines) ok={ok}", flush=True)
    return ok


def main():
    cu = sys.argv[1] if len(sys.argv) > 1 else ""   # wave sub-arm divisor ("48"=runtime default, "96"=physical)
    env1 = [f"R3D_SPLIT_POLICY=1"] + ([f"R3D_SPLIT_CU={cu}"] if cu else [])
    arms = [("q46_pol0", ["R3D_SPLIT_POLICY=0"]), ("q46_wave", env1)]
    st = sh("docker", "ps", "--format", "{{.Names}} {{.Status}}").stdout
    assert "spec9g " not in st, "spec9g must be stopped for the serve A/B"
    for n, _ in arms:
        sh("docker", "rm", "-f", n)
    for n, e in arms:
        ensure_arm_container(n, e)
    order = [arms[0], arms[1]] * 3 + [(arms[0][0], arms[0][1])]  # trailing baseline confirm (promotion-ceremony law)
    for rnd, (name, envs) in enumerate(order, 1):
        other = "q46_wave" if name == "q46_pol0" else "q46_pol0"
        sh("docker", "stop", "-t", "20", other)
        ensure_arm_container(name, envs)
        if not start_arm(name):
            print(f"!! {name} cannot start", flush=True)
            sys.exit(1)
        if not ready():
            print(f"!! {name} not tokenize-ready", flush=True)
            sys.exit(1)
        if not settle(name):
            print(f"!! {name} never settled", flush=True)
            sys.exit(1)
        ok = liveness(name, envs[0])
        rate, line = decode_round(name)
        tag = "CONFIRM-A" if rnd == len(order) else f"r{rnd}"
        pol = "p0" if "POLICY=0" in envs[0] else ("wave-cu96" if cu == "96" else "wave-cu48")
        print(f"DEC arm={pol} {tag} tok/s={rate} {'LIVENESS-FAIL' if not ok else ''}",
              flush=True)
        print(f"  {line}", flush=True)
    for n, _ in arms:
        sh("docker", "rm", "-f", n)
    print("== serve A/B DONE ==", flush=True)


if __name__ == "__main__":
    main()
