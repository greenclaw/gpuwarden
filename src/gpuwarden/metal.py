"""gwctl metal — serve a model on a GPU box you own (the dedicated-card half of gpuwarden).

cloud.sh rents a pod; this module serves the same MODELS_DIR/<label>/serve.env on local hardware.
The separation that matters: serve.env is the *invariant* (what to serve — image digest, weights
revision, parsers, context), while the deployment target is just a renderer over it — docker
compose today, a Kubernetes manifest when the infra moves. `verify` needs only a URL, so the
acceptance layer survives the move untouched.

    gwctl provision [--apply]        # diagnose (or install) driver/docker/toolkit on this box
    gwctl render <label> [--target compose|k8s] [--stdout]
    gwctl serve  <label> [--replace] # render + up + wait healthy + verify (--replace: stop the serve
                                     #   already holding the card — one GPU, one serve)
    gwctl verify <label | --url URL> [--container NAME]
                                     # health, /v1/models, engine facts + flags, tool-calling acceptance

Hard-won rules this module encodes (each cost a real incident):
  * Blackwell (GB2xx) initializes ONLY with the open kernel module. The proprietary one binds and
    looks healthy while nvidia-smi says "No devices were found" (VBIOS ??.??, firmware N/A).
  * vLLM does NOT auto-enable prefix caching for hybrid-Mamba models (Qwen3.6-A3B & friends);
    omit --enable-prefix-caching and you silently serve a config no benchmark was taken on.
  * An empty VLLM_API_KEY means an unauthenticated OpenAI endpoint — refuse, never hope.
  * A driver upgrade under a running serve is invisible until the next restart: the loaded module
    stays old, userspace goes new, NVML refuses to init, and the stale CDI spec still names the old
    libraries. provision checks module-vs-userspace, CDI mount sources, and that the driver is held.
"""
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ---------- serve.env → vLLM invocation (shared by every render target) ----------

def read_serve_env(c: dict, label: str) -> dict:
    from .cli import models_dir
    path = models_dir(c) / label / "serve.env"
    if not path.is_file():
        sys.exit(f"no serve.env for label '{label}' under {path.parent.parent}")
    e = {}
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            e[k.strip()] = v.strip().strip('"').strip("'")
    for req in ("IMAGE", "MODEL", "SERVED_NAME", "TOOL_PARSER", "MAXLEN"):
        if not e.get(req):
            sys.exit(f"{path}: serve.env must set {req}")
    if "@sha256:" not in e["IMAGE"]:
        print("WARN: IMAGE is not pinned by digest — a tag can be re-pushed under you", file=sys.stderr)
    e["_label"], e["_path"] = label, path
    return e


def vllm_args(e: dict) -> list:
    """The one authoritative arg builder — parity with cloud.sh's ARGS line. Toggles default ON
    because every recorded benchmark ran with them; set PREFIX_CACHING=0 / TRUST_REMOTE_CODE=0
    in serve.env to opt out deliberately (and re-bench)."""
    a = ["--model", e["MODEL"]]
    if e.get("REVISION"):
        a += ["--revision", e["REVISION"]]
    if e.get("TOKENIZER"):
        a += ["--tokenizer", e["TOKENIZER"]]
    a += ["--served-model-name", e["SERVED_NAME"],
          "--enable-auto-tool-choice", "--tool-call-parser", e["TOOL_PARSER"]]
    if e.get("REASONING_PARSER") and e["REASONING_PARSER"] != "none":
        a += ["--reasoning-parser", e["REASONING_PARSER"]]
    if e.get("TRUST_REMOTE_CODE", "1") != "0":
        a += ["--trust-remote-code"]
    if e.get("PREFIX_CACHING", "1") != "0":
        a += ["--enable-prefix-caching"]
    a += ["--max-model-len", e["MAXLEN"]]
    a += shlex.split(e.get("EXTRA_ARGS", ""))
    a += ["--host", "0.0.0.0", "--port", e.get("PORT", "8000")]
    return a


def render_compose(e: dict) -> str:
    args = "\n".join(f"      {x}" for x in vllm_args(e))
    cache = e.get("HF_CACHE", "~/.cache/huggingface").replace("~", "${HOME}")
    return f"""# rendered by `gwctl render {e['_label']}` from serve.env — regenerate, don't hand-edit
# VLLM_API_KEY must be in the environment at `up` time (gwctl serve injects it from KEYS_FILE);
# the ${{...:?}} guard makes a bare `docker compose up` fail closed instead of serving keyless.
services:
  vllm:
    image: {e['IMAGE']}
    container_name: {container_name(e)}
    restart: unless-stopped
    network_mode: host
    ipc: host
    environment:
      VLLM_API_KEY: ${{VLLM_API_KEY:?refusing to serve an unauthenticated endpoint}}
    volumes:
      - {cache}:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: >
{args}
"""


def render_k8s(e: dict) -> str:
    """A reviewed starting point, not a fire-and-forget: storage class, GPU operator setup and
    ingress are cluster-specific. The args block is the same authoritative vllm_args()."""
    args = "\n".join(f"            - \"{x}\"" for x in vllm_args(e))
    port = e.get("PORT", "8000")
    name = f"gw-{e['_label']}"
    return f"""# rendered by `gwctl render {e['_label']} --target k8s` — STARTING POINT, review before applying.
# Prereqs: NVIDIA GPU operator (or device plugin); a Secret with the endpoint key:
#   kubectl create secret generic {name}-key --from-literal=VLLM_API_KEY=...
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
spec:
  replicas: 1
  strategy: {{type: Recreate}}     # one GPU, one pod — never two claimants during rollout
  selector: {{matchLabels: {{app: {name}}}}}
  template:
    metadata:
      labels: {{app: {name}}}
    spec:
      containers:
        - name: vllm
          image: {e['IMAGE']}
          args:
{args}
          env:
            - name: VLLM_API_KEY
              valueFrom: {{secretKeyRef: {{name: {name}-key, key: VLLM_API_KEY}}}}
          ports:
            - containerPort: {port}
          resources:
            limits: {{nvidia.com/gpu: 1}}
          volumeMounts:
            - {{name: hf-cache, mountPath: /root/.cache/huggingface}}
            - {{name: shm, mountPath: /dev/shm}}
          readinessProbe:
            httpGet: {{path: /health, port: {port}}}
            periodSeconds: 15
            failureThreshold: 120   # cold start = weights + compile; give it 30 min
      volumes:
        - name: hf-cache
          persistentVolumeClaim: {{claimName: {name}-hf-cache}}   # create one; weights re-download without it
        - name: shm
          emptyDir: {{medium: Memory, sizeLimit: 16Gi}}
---
apiVersion: v1
kind: Service
metadata:
  name: {name}
spec:
  selector: {{app: {name}}}
  ports:
    - port: {port}
      targetPort: {port}
"""


def cmd_render(c, a) -> int:
    e = read_serve_env(c, a.label)
    text = render_k8s(e) if a.target == "k8s" else render_compose(e)
    if a.stdout:
        print(text, end="")
        return 0
    out = e["_path"].with_name("k8s.yaml" if a.target == "k8s" else "compose.yaml")
    out.write_text(text)
    print(f"rendered -> {out}")
    return 0


# ---------- box-state parsers (pure: every input is text a real box produced) ----------

def module_version(proc_version: str):
    """Version of the kernel module actually loaded, from /proc/driver/nvidia/version."""
    m = re.search(r"Kernel Module\b.*?\s(\d+\.\d+(?:\.\d+)?)\s", proc_version)
    return m.group(1) if m else None


def driver_drift(module_ver, nvml_libs: list):
    """Userspace upgraded under a loaded module: NVML refuses to init, and no container can claim
    the GPU — yet a serve that already held the card keeps running, so the drift stays invisible
    until the next restart or reboot. None when there is no evidence either way."""
    lib_vers = sorted({m.group(1) for p in nvml_libs
                       if (m := re.search(r"libnvidia-ml\.so\.(\d+\.\d+(?:\.\d+)?)$", p))})
    if not module_ver or not lib_vers or lib_vers == [module_ver]:
        return None
    return f"loaded kernel module {module_ver}, userspace driver {', '.join(lib_vers)}"


def stale_cdi_paths(spec_text: str, exists=os.path.exists) -> list:
    """Mount sources a CDI spec names but the host no longer has. The spec is a snapshot taken
    at generation time: a driver upgrade renames every library (libEGL_nvidia.so.<old>), and a
    stopped nvidia-persistenced removes its socket — either way container create fails."""
    paths = re.findall(r"hostPath:\s*(\S+)", spec_text)
    return [p for p in dict.fromkeys(paths) if not exists(p)]


def serve_claimants(running: list, own: str) -> list:
    """Other vLLM serves on the card. One GPU = one serve: two would fight over VRAM and the
    port. Monitoring that merely reads the GPU (dcgm-exporter) is not a rival."""
    return [c["name"] for c in running
            if c.get("gpu") and "vllm" in c.get("image", "") and c["name"] != own]


def container_crashed(state: str):
    """`docker inspect -f '{{.State.Status}} {{.RestartCount}}'` -> why startup failed, or None.
    With restart: unless-stopped a dying engine does not stay dead — it loops quietly, so a
    restart count above zero during startup is already the failure."""
    parts = state.split()
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    status, restarts = parts[0], int(parts[1])
    if status in ("exited", "dead"):
        return "container exited"
    if restarts > 0:
        return f"container restarted {restarts}x during startup"
    return None


def container_name(e: dict) -> str:
    """gw-<label>, unless serve.env pins CONTAINER_NAME — how an existing serve moves under gwctl
    without renaming what logs, dashboards and runbooks already know it as."""
    return e.get("CONTAINER_NAME") or f"gw-{e['_label']}"


def compose_project(label: str) -> str:
    """The compose project gwctl uses for a label (docker's own normalisation, made explicit)."""
    return re.sub(r"[^a-z0-9_-]", "", label.lower())


def adopt_rename(own: str, existing_project, our_project: str, stamp: str):
    """A container already holding our name but created by ANOTHER compose project (a hand-made
    /opt/... compose) blocks `compose up`. Rename it out of the way — it stays restorable."""
    if not existing_project or existing_project == our_project:
        return None
    return f"{own}-replaced-{stamp}"


def rollback_plan(own: str, replaced: list, renamed=None) -> list:
    """What a failed serve must undo: stop its own container (restart: unless-stopped would
    otherwise loop it forever) and start again every serve that --replace stopped for it. An
    adopted container gets its original name back — which first means freeing that name."""
    renamed = renamed or {}
    plan = [["docker", "stop", own]]
    if renamed:
        plan.append(["docker", "rm", own])
    for r in replaced:
        if r in renamed:
            plan += [["docker", "rename", r, renamed[r]], ["docker", "start", renamed[r]]]
        else:
            plan.append(["docker", "start", r])
    return plan


def driver_packages(dpkg_out: str) -> list:
    """Installed NVIDIA driver packages from `dpkg-query -W -f='${db:Status-Abbrev}|${Package}'`.
    Installed = second status letter 'i', whatever the wanted state: a HELD package reports 'hi',
    and filtering on 'ii' made the hold check vanish right after --apply had held everything.
    The container toolkit moves on its own schedule and is not pinned with the driver."""
    out = []
    for ln in dpkg_out.splitlines():
        abbrev, _, pkg = ln.partition("|")
        abbrev, pkg = abbrev.strip(), pkg.strip()
        if len(abbrev) >= 2 and abbrev[1] == "i" and pkg and "container" not in pkg:
            out.append(pkg)
    return out


def engine_facts(log: str) -> dict:
    """What the engine actually chose, from its startup log — the flags on the command line
    are intent; these lines are the outcome (dtype, block size, kernel fallbacks)."""
    f = {}
    pats = {
        "vllm_version": (r"LLM engine \(v([\d.]+\w*)\)", str),
        "kv_cache_tokens": (r"GPU KV cache size: ([\d,]+) tokens", lambda s: int(s.replace(",", ""))),
        "kv_cache_dtype": (r"kv_cache_dtype=(\w+)", str),
        "attention_block_size": (r"Setting attention block size to (\d+) tokens", int),
        "mamba_cache_mode": (r"Mamba cache mode is set to '(\w+)'", str),
        "nvfp4_moe_backend": (r"Using '(\w+)' NvFp4 MoE backend", str),
        "enable_prefix_caching": (r"enable_prefix_caching=(\w+)", str),
        "weights_gib": (r"Model loading took ([\d.]+) GiB", float),
        "kv_cache_gib": (r"Available KV cache memory: ([\d.]+) GiB", float),
    }
    for key, (pat, conv) in pats.items():
        hits = re.findall(pat, log)
        if hits:
            f[key] = conv(hits[-1])            # last start wins if the log spans restarts
    m = re.findall(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+x)", log)
    if m:
        f["max_concurrency"] = f"{m[-1][1]} @ {m[-1][0]}"
    if "does not have native support for FP4" in log:
        f["fp4_native"] = False
    return f


# ---------- serve + verify ----------

def _vllm_key(c: dict) -> str:
    from .cli import load_keys
    env = load_keys(c)
    key = env.get("VLLM_API_KEY") or env.get("VLLM_POD_KEY") or ""
    if not key.strip():
        sys.exit("FATAL: no VLLM_API_KEY / VLLM_POD_KEY in env or KEYS_FILE — "
                 "refusing to serve an unauthenticated endpoint")
    return key.strip()


def _get(url: str, key: str = "", timeout: int = 8):
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as ex:
        return ex.code, ""
    except OSError:
        return 0, ""


def running_containers() -> list:
    r = subprocess.run(["bash", "-c", "docker ps -q | xargs -r docker inspect --format "
                        "'{{.Name}}\t{{.Config.Image}}\t{{json .HostConfig.DeviceRequests}}'"],
                       capture_output=True, text=True)
    out = []
    for ln in (r.stdout or "").splitlines():
        name, image, dev = (ln.split("\t", 2) + ["", ""])[:3]
        out.append({"name": name.lstrip("/"), "image": image, "gpu": "gpu" in dev})
    return out


def _rollback(c, own: str, replaced: list, base_port: str = "8000", renamed=None) -> int:
    """Undo a failed serve and report whether what it replaced is serving again."""
    from .cli import log
    for cmd in rollback_plan(own, replaced, renamed):
        rc = subprocess.run(cmd, capture_output=True, text=True).returncode
        log(c, f"serve: rollback: {' '.join(cmd[1:])} -> rc={rc}")
    if replaced:
        base = f"http://127.0.0.1:{base_port}"
        for _ in range(60):                    # 10 min: a restarted serve re-warms its engine
            if _get(base + "/health")[0] == 200:
                log(c, f"serve: rollback: {[(renamed or {}).get(r, r) for r in replaced]} healthy again at {base}")
                break
            time.sleep(10)
        else:
            log(c, f"serve: rollback: {replaced} started but NOT healthy after 10 min — CHECK NOW")
    return 1


def cmd_serve(c, a) -> int:
    from .cli import log
    e = read_serve_env(c, a.label)
    key = _vllm_key(c)
    own, project = container_name(e), compose_project(a.label)
    renamed = {}
    st = subprocess.run(["docker", "inspect", "-f",
                         '{{index .Config.Labels "com.docker.compose.project"}}', own],
                        capture_output=True, text=True)
    theirs = st.stdout.strip() if st.returncode == 0 else None
    if theirs is not None and theirs != project:
        if not getattr(a, "replace", False):
            log(c, f"serve: REFUSING — '{own}' exists but belongs to compose project '{theirs or '?'}'; "
                   "re-run with --replace to take it over (it is renamed, not deleted)")
            return 1
        new = adopt_rename(own, theirs or "(none)", project, time.strftime("%Y%m%dT%H%M%S"))
        subprocess.run(["docker", "rename", own, new], capture_output=True, text=True)
        renamed[new] = own
        log(c, f"serve: adopting the name '{own}': existing container renamed to '{new}'")
    rivals = serve_claimants(running_containers(), own=own)
    if rivals and not getattr(a, "replace", False):
        log(c, f"serve: REFUSING — the GPU already runs {rivals}; one card = one serve. "
               "Re-run with --replace to stop them first.")
        return 1
    replaced = []
    for r in rivals:
        log(c, f"serve: --replace: stopping {r} (restored automatically if {own} fails to come up)")
        if subprocess.run(["docker", "stop", r], capture_output=True, text=True).returncode == 0:
            replaced.append(r)
    compose = e["_path"].with_name("compose.yaml")
    compose.write_text(render_compose(e))
    log(c, f"serve: {a.label} — compose rendered, starting (cold start = pull + weights + compile)")
    up = ["docker", "compose", "-f", str(compose), "-p", project, "up", "-d"]
    if getattr(a, "recreate", False):          # fresh engine = empty prefix cache, same config
        up.append("--force-recreate")
    r = subprocess.run(up,
                       env={**os.environ, "VLLM_API_KEY": key}, capture_output=True, text=True)
    if r.returncode != 0:
        log(c, f"serve: docker compose FAILED rc={r.returncode}\n{(r.stderr or '').strip()[-600:]}")
        return _rollback(c, own, replaced, base_port=e.get("PORT", "8000"), renamed=renamed)
    base = f"http://127.0.0.1:{e.get('PORT', '8000')}"
    for i in range(360):                       # 360 x 10s = 60 min ceiling
        code, _ = _get(base + "/health")
        if code == 200:
            break
        st = subprocess.run(["docker", "inspect", "-f", "{{.State.Status}} {{.RestartCount}}", own],
                            capture_output=True, text=True).stdout.strip()
        why = container_crashed(st)
        if why:
            tail = subprocess.run(["docker", "logs", "--tail", "30", own],
                                  capture_output=True, text=True)
            log(c, f"serve: FAILED — {why}; last log lines:\n"
                   f"{((tail.stdout or '') + (tail.stderr or '')).strip()[-2500:]}")
            return _rollback(c, own, replaced, base_port=e.get("PORT", "8000"), renamed=renamed)
        time.sleep(10)
    else:
        log(c, f"serve: NOT healthy after 60 min — docker logs {own}")
        return _rollback(c, own, replaced, base_port=e.get("PORT", "8000"), renamed=renamed)
    log(c, f"serve: healthy at {base}")
    return _verify(c, e, base, key, container=own)


def cmd_verify(c, a) -> int:
    container = getattr(a, "container", None)
    if a.url:
        return _verify(c, None, a.url.rstrip("/"), _vllm_key(c), container=container)
    if not a.label:
        sys.exit("verify: give a <label> or --url")
    e = read_serve_env(c, a.label)
    return _verify(c, e, f"http://127.0.0.1:{e.get('PORT', '8000')}", _vllm_key(c),
                   container=container or container_name(e))


def _verify(c, e, base: str, key: str, container=None) -> int:
    """Target-agnostic acceptance: works against compose, k8s, or a rented pod — anything with a URL.
    Next to the container it additionally reports what the ENGINE chose and proves it runs the
    flags the config asked for, because 'the flag was on the command line' is not 'the engine
    enabled it' (hybrid-Mamba APC). `container` covers serves not started by gwctl."""
    from .cli import log
    problems = []
    code, body = _get(base + "/v1/models", key)
    if code != 200:
        log(c, f"verify: FAIL — /v1/models {code or 'unreachable'}")
        return 1
    served = [m["id"] for m in json.loads(body).get("data", [])]
    log(c, f"verify: models {served}")
    if e and e["SERVED_NAME"] not in served:
        problems.append(f"served names {served} lack SERVED_NAME {e['SERVED_NAME']}")

    if container:  # engine facts + flag check, only meaningful next to the container
        r = subprocess.run(["docker", "logs", container], capture_output=True, text=True)
        facts = engine_facts((r.stdout or "") + (r.stderr or ""))
        for k, v in facts.items():
            log(c, f"verify: engine {k} = {v}")
        if facts.get("fp4_native") is False:
            log(c, "verify: NOTE — FP4 runs on the weight-only Marlin fallback this vLLM chose; "
                   "a throughput limit, not a correctness one")
        if e and "enable_prefix_caching" in facts:
            want = e.get("PREFIX_CACHING", "1") != "0"
            if (facts["enable_prefix_caching"] == "True") != want:
                problems.append(f"engine enable_prefix_caching={facts['enable_prefix_caching']} but "
                                f"serve.env wants {want} (vLLM does not auto-enable APC for "
                                "hybrid-Mamba — check the flag)")

    model = e["SERVED_NAME"] if e else served[0]
    payload = json.dumps({
        "model": model, "max_tokens": 1024,   # thinking models burn small budgets inside reasoning
        "messages": [{"role": "user",
                      "content": "Client ACME has invoices 17250, 9800 and 23450. What is the total?"}],
        "tools": [{"type": "function", "function": {
            "name": "sum_invoices", "description": "Sum invoice amounts for a client",
            "parameters": {"type": "object", "properties": {
                "client": {"type": "string"},
                "amounts": {"type": "array", "items": {"type": "number"}}},
                "required": ["client", "amounts"]}}}]})
    req = urllib.request.Request(base + "/v1/chat/completions", data=payload.encode(),
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            choice = json.loads(r.read())["choices"][0]
        calls = choice["message"].get("tool_calls") or []
        args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
        if choice.get("finish_reason") != "tool_calls" or \
                sorted(args.get("amounts", [])) != [9800, 17250, 23450]:
            problems.append(f"tool-calling acceptance failed: finish={choice.get('finish_reason')} args={args}")
        else:
            log(c, "verify: tool-calling acceptance PASSED")
    except (OSError, KeyError, IndexError, ValueError) as ex:
        problems.append(f"tool-calling acceptance errored: {ex}")

    for p in problems:
        log(c, f"verify: PROBLEM — {p}")
    log(c, f"verify: {'PASSED' if not problems else 'FAILED'} ({len(problems)} problem(s))")
    return 1 if problems else 0


# ---------- provision ----------

def _run(cmd: str) -> tuple:
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()


def cmd_provision(c, a) -> int:
    """Diagnose the box. --apply executes the fix steps (Ubuntu/Debian, needs sudo) — the exact
    sequence proven on a live bring-up; on other distros it prints them for you to translate."""
    checks, fixes = [], []

    rc, out, _ = _run("lspci | grep -i 'nvidia'")
    checks.append(("GPU on PCI bus", rc == 0, out.splitlines()[0] if out else "no NVIDIA device found"))
    gpu_on_bus, blackwell = rc == 0, "GB2" in out or "Blackwell" in out

    rc, out, _ = _run("nvidia-smi -L")
    smi_ok = rc == 0 and "GPU 0" in out
    checks.append(("driver initializes GPU", smi_ok, out.splitlines()[0] if out else "nvidia-smi failed"))
    if gpu_on_bus and not smi_ok:
        rc, lic, _ = _run("modinfo nvidia 2>/dev/null | sed -n 's/^license: *//p'")
        if lic and "GPL" not in lic:
            note = ("PROPRIETARY kernel module on a Blackwell card — it binds but cannot init "
                    "(telltale: VBIOS ??.?? in /proc/driver/nvidia). Fix: the -open driver."
                    if blackwell else "proprietary kernel module; consider the -open variant")
            checks.append(("open kernel module", False, note))
            _, ver, _ = _run("modinfo nvidia 2>/dev/null | sed -n 's/^version: *\\([0-9]*\\).*/\\1/p'")
            fixes.append(f"sudo apt-get install -y nvidia-driver-{ver or '<major>'}-server-open && sudo reboot")

    # Userspace upgraded under the loaded module: a running serve keeps working on the old
    # module, so nothing looks wrong until the next restart — which then cannot claim the GPU.
    manual = []
    _, proc, _ = _run("cat /proc/driver/nvidia/version 2>/dev/null")
    _, libs, _ = _run("ls -1 /usr/lib/*/libnvidia-ml.so.*.* /usr/lib64/libnvidia-ml.so.*.* 2>/dev/null")
    drift = driver_drift(module_version(proc), libs.splitlines())
    checks.append(("loaded module matches driver", drift is None, drift or ""))
    if drift:
        holders = [x["name"] for x in running_containers() if x["gpu"]]
        manual.append(f"stop everything holding the GPU ({', '.join(holders) or 'check `lsof /dev/nvidia*`'}), "
                      "then: sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && "
                      "sudo modprobe nvidia && sudo modprobe nvidia_uvm   (or: sudo reboot) — "
                      "then re-run provision: the CDI spec needs regenerating after the swap")

    # CDI spec = snapshot of library names at generation time; stale after a driver upgrade.
    rc, specs, _ = _run("ls -1 /etc/cdi/*.yaml /var/run/cdi/*.yaml 2>/dev/null")
    for spec in dict.fromkeys(os.path.realpath(s) for s in specs.splitlines()):
        try:
            with open(spec) as fh:
                missing = stale_cdi_paths(fh.read())
        except OSError:
            continue
        checks.append((f"CDI spec current ({spec})", not missing,
                       f"{len(missing)} mount source(s) gone, e.g. {missing[0]}" if missing else ""))
        if any(p.startswith("/run/nvidia-persistenced") for p in missing):
            fixes.append("sudo systemctl start nvidia-persistenced")
        if any(not p.startswith("/run/nvidia-persistenced") for p in missing):
            fixes.append(f"sudo nvidia-ctk cdi generate --output={spec}")

    # Pin the driver like everything else: unattended-upgrades bumping it under a live serve is
    # exactly how the drift above happens.
    rc, pkgs, _ = _run("dpkg-query -W -f='${db:Status-Abbrev}|${Package}\\n' "
                       "'nvidia-*' 'libnvidia-*' 'xserver-xorg-video-nvidia-*' 2>/dev/null")
    driver_pkgs = driver_packages(pkgs)
    if driver_pkgs:
        _, held, _ = _run("apt-mark showhold 2>/dev/null")
        unheld = sorted(set(driver_pkgs) - set(held.split()))
        checks.append(("driver packages held", not unheld,
                       f"{len(unheld)} can be upgraded silently" if unheld else ""))
        if unheld:
            fixes.append("sudo apt-mark hold " + " ".join(unheld))

    rc, _, _ = _run("command -v docker")
    checks.append(("docker present", rc == 0, "" if rc == 0 else "not installed"))
    if rc != 0:
        fixes.append("curl -fsSL https://get.docker.com | sudo sh")

    rc, out, _ = _run("docker info --format '{{json .Runtimes}}' 2>/dev/null")
    has_nvidia_rt = rc == 0 and "nvidia" in out
    checks.append(("nvidia container runtime", has_nvidia_rt, "" if has_nvidia_rt else "toolkit missing/unconfigured"))
    if not has_nvidia_rt:
        fixes.append("sudo apt-get install -y nvidia-container-toolkit && "
                     "sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker")

    width = max(len(n) for n, _, _ in checks)
    for name, ok, note in checks:
        print(f"  {'OK ' if ok else 'FAIL'} {name:<{width}}  {note}")
    if not fixes and not manual:
        print("provision: box looks ready — try `gwctl serve <label>`")
        return 0
    if manual:
        print("\nmanual steps (disruptive — --apply will not run these for you):")
        for m in manual:
            print(f"  $ {m}")
    if fixes:
        print("\nfix steps" + (" (running with --apply):" if a.apply else " (re-run with --apply to execute):"))
    for f in fixes:
        print(f"  $ {f}")
        if a.apply:
            rc = subprocess.run(["bash", "-c", f]).returncode
            if rc != 0:
                print(f"provision: step failed rc={rc} — stopping here")
                return rc
    return 0 if (a.apply and not manual) else 1
