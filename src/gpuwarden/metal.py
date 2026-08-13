"""gwctl metal — serve a model on a GPU box you own (the dedicated-card half of gpuwarden).

cloud.sh rents a pod; this module serves the same MODELS_DIR/<label>/serve.env on local hardware.
The separation that matters: serve.env is the *invariant* (what to serve — image digest, weights
revision, parsers, context), while the deployment target is just a renderer over it — docker
compose today, a Kubernetes manifest when the infra moves. `verify` needs only a URL, so the
acceptance layer survives the move untouched.

    gwctl provision [--apply]        # diagnose (or install) driver/docker/toolkit on this box
    gwctl render <label> [--target compose|k8s] [--stdout]
    gwctl serve  <label>             # render + up + wait healthy + verify
    gwctl verify <label | --url URL> # health, /v1/models, engine flags, tool-calling acceptance

Hard-won rules this module encodes (each cost a real incident):
  * Blackwell (GB2xx) initializes ONLY with the open kernel module. The proprietary one binds and
    looks healthy while nvidia-smi says "No devices were found" (VBIOS ??.??, firmware N/A).
  * vLLM does NOT auto-enable prefix caching for hybrid-Mamba models (Qwen3.6-A3B & friends);
    omit --enable-prefix-caching and you silently serve a config no benchmark was taken on.
  * An empty VLLM_API_KEY means an unauthenticated OpenAI endpoint — refuse, never hope.
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
    container_name: gw-{e['_label']}
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


def cmd_serve(c, a) -> int:
    from .cli import log
    e = read_serve_env(c, a.label)
    key = _vllm_key(c)
    compose = e["_path"].with_name("compose.yaml")
    compose.write_text(render_compose(e))
    log(c, f"serve: {a.label} — compose rendered, starting (cold start = pull + weights + compile)")
    r = subprocess.run(["docker", "compose", "-f", str(compose), "up", "-d"],
                       env={**os.environ, "VLLM_API_KEY": key}, capture_output=True, text=True)
    if r.returncode != 0:
        log(c, f"serve: docker compose FAILED rc={r.returncode}\n{(r.stderr or '').strip()[-600:]}")
        return 1
    base = f"http://127.0.0.1:{e.get('PORT', '8000')}"
    for i in range(120):                       # 120 x 30s = 60 min ceiling
        code, _ = _get(base + "/health")
        if code == 200:
            break
        time.sleep(30)
    else:
        log(c, f"serve: NOT healthy after 60 min — docker logs gw-{a.label}")
        return 1
    log(c, f"serve: healthy at {base}")
    return _verify(c, e, base, key)


def cmd_verify(c, a) -> int:
    if a.url:
        return _verify(c, None, a.url.rstrip("/"), _vllm_key(c))
    if not a.label:
        sys.exit("verify: give a <label> or --url")
    e = read_serve_env(c, a.label)
    return _verify(c, e, f"http://127.0.0.1:{e.get('PORT', '8000')}", _vllm_key(c))


def _verify(c, e, base: str, key: str) -> int:
    """Target-agnostic acceptance: works against compose, k8s, or a rented pod — anything with a URL.
    With a local serve.env (e) it additionally proves the ENGINE runs the flags the config asked for,
    because 'the flag was on the command line' is not 'the engine enabled it' (hybrid-Mamba APC)."""
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

    if e:  # engine-flag check, only meaningful next to the container
        r = subprocess.run(["docker", "logs", f"gw-{e['_label']}"], capture_output=True, text=True)
        m = re.findall(r"enable_prefix_caching=(\w+)", (r.stdout or "") + (r.stderr or ""))
        want = e.get("PREFIX_CACHING", "1") != "0"
        if m and (m[-1] == "True") != want:
            problems.append(f"engine enable_prefix_caching={m[-1]} but serve.env wants {want} "
                            "(vLLM does not auto-enable APC for hybrid-Mamba — check the flag)")

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
    if not fixes:
        print("provision: box looks ready — try `gwctl serve <label>`")
        return 0
    print("\nfix steps" + (" (running with --apply):" if a.apply else " (re-run with --apply to execute):"))
    for f in fixes:
        print(f"  $ {f}")
        if a.apply:
            rc = subprocess.run(["bash", "-c", f]).returncode
            if rc != 0:
                print(f"provision: step failed rc={rc} — stopping here")
                return rc
    return 0 if a.apply else 1
