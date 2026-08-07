#!/usr/bin/env python3
"""gwctl — owns RunPod pod lifecycle on the always-on dev server (single source of truth).

Talks to the RunPod HTTP API (runpodctl + GraphQL); it never touches a local Docker daemon — the pods
run on RunPod's infrastructure. Runs ON THE SERVER; the laptop is a thin ssh client (remote.sh).
Keys live on the server only — never on the laptop, never in a repo (see KEYS_FILE below).

Two modes, one mechanism (cron):
  * regular  — `up` on UP_AT (workdays per DAYS), `down` on DOWN_AT EVERY day
  * one-shot — just run `up` any time; the daily `down` + late `sweep` guarantee teardown, so a
               manual/weekend pod can never burn overnight. That's why no at/systemd-timer is needed.

    gwctl view                     # config, cron state, rendered lines, live pods + balance
    gwctl set --up 11:00 --down 18:30 --model <label> --days mon-fri --models-dir ~/models
    gwctl install | uninstall      # manage the marker-delimited crontab block
    gwctl up | down                # the actions cron calls (also fine by hand)
"""
import argparse, os, re, subprocess, sys, time
from pathlib import Path

CLOUD_SH = Path(__file__).resolve().with_name("cloud.sh")   # ships inside the installed package
CONF_DIR = Path(os.path.expanduser(os.environ.get("GPUWARDEN_CONF_DIR", "~/.config/gpuwarden")))
CONF = CONF_DIR / "scheduler.conf"                          # state lives outside any repo checkout
BEGIN, END = "# BEGIN gpuwarden", "# END gpuwarden"
DEFAULTS = {
    # Directory holding <label>/serve.env for each servable model. The tool installs separately
    # from your model configs, so this is configuration, not a relative path.
    "MODELS_DIR": "~/gpuwarden/models",
    "MODEL_LABEL": "example",
    # Optional post-health hook, run with BASE_URL / SERVED_MODEL / LABEL / VLLM_API_KEY in its env —
    # use it to absorb the cold-start spike before real traffic arrives. Empty = skip.
    "WARMUP_CMD": "",
    "UP_AT": "11:00",          # HH:MM, local time on this host
    "DOWN_AT": "18:30",
    "SWEEP_AT": "23:30",       # last-resort teardown, every day
    "DAYS": "mon-fri",         # days the scheduled UP runs (down/sweep always run daily)
    "TZ": "Europe/Moscow",
    "MIN_BALANCE": "2.00",     # skip `up` below this — don't raise a pod that dies mid-day
    # Keys file (RUNPOD_API_KEY, VLLM_POD_KEY) — never in a repo, never written by tooling.
    "KEYS_FILE": "~/.config/gpuwarden/keys.env",
    "LOG_FILE": "~/.local/state/gpuwarden.log",
}


def models_dir(c: dict) -> Path:
    return Path(os.path.expanduser(c["MODELS_DIR"])).resolve()
DAY_CRON = {"mon-fri": "1-5", "every": "*"}


def load_conf() -> dict:
    c = dict(DEFAULTS)
    if CONF.exists():
        for ln in CONF.read_text().splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, v = ln.split("=", 1)
                c[k.strip()] = v.strip().strip('"').strip("'")
    return c


def save_conf(c: dict) -> None:
    body = ["# gwctl config — NON-SECRET only (keys live in KEYS_FILE).",
            "# Edit via: gwctl set --up HH:MM --down HH:MM --model <label> --days mon-fri|every", ""]
    body += [f"{k}={c[k]}" for k in DEFAULTS]
    CONF_DIR.mkdir(parents=True, exist_ok=True)
    CONF.write_text("\n".join(body) + "\n")


def load_keys(c: dict) -> dict:
    """Env for cloud.sh: the real environment wins, otherwise read KEYS_FILE."""
    env = dict(os.environ)
    kf = Path(os.path.expanduser(c["KEYS_FILE"]))
    if not env.get("RUNPOD_API_KEY") and kf.exists():
        for ln in kf.read_text().splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, v = ln.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return env


def log(c: dict, msg: str) -> None:
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    try:                                    # 0600: the log can carry endpoint output
        p = Path(os.path.expanduser(c["LOG_FILE"]))
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as f:
            f.write(line + "\n")
    except OSError as e:                    # never let logging abort a teardown
        print(f"(log write failed: {e})", flush=True)


def cloud(c: dict, *args: str, timeout: int = 3900) -> subprocess.CompletedProcess:
    # default outlives cloud.sh's own health wait (120 polls x (curl 8s + sleep 20s) ~= 3400s),
    # otherwise a normal slow community pull raises TimeoutExpired with the pod already live.
    env = load_keys(c)
    env["MODELS_DIR"] = str(models_dir(c))   # cloud.sh resolves <label>/serve.env against this
    env["WARMUP_CMD"] = c.get("WARMUP_CMD", "")
    return subprocess.run(["bash", str(CLOUD_SH), *args], cwd=models_dir(c),
                          env=env, capture_output=True, text=True, timeout=timeout)


def status(c: dict):
    """-> (state, balance_or_None, raw); state is "pods" | "none" | "unknown".

    UNKNOWN is never collapsed into "none". Failing to *observe* must not read as observing zero:
    cloud.sh runs under `set -e`, so an auth failure or an API 5xx yields empty stdout,
    which is byte-identical to a healthy empty account. Consumers fail closed on unknown — a skipped
    `up` costs nothing, whereas a false "no pods" either duplicates a billing pod or declares a
    teardown VERIFIED while the pod keeps running.
    """
    try:
        r = cloud(c, "status", timeout=180)
    except subprocess.TimeoutExpired:
        return ("unknown", None, "cloud.sh status timed out")
    raw = ((r.stdout or "") + (r.stderr or "")).strip()
    m = re.search(r"balance \$(-?[0-9.]+)", raw)
    bal = float(m.group(1)) if m else None
    # cloud.sh prints the balance line LAST; its absence proves the script died before finishing.
    if r.returncode != 0 or bal is None:
        return ("unknown", bal, raw)
    return ("none" if "no pods" in raw else "pods", bal, raw)


def pod_hours(c: dict) -> int:
    """Hours for cloud.sh --terminate-after: must OUTLIVE the scheduled window, never cut it short.
    Spans UP_AT..SWEEP_AT (the last teardown of the day) plus an hour of margin."""
    uh, um = (int(x) for x in c["UP_AT"].split(":"))
    sh_, sm = (int(x) for x in c["SWEEP_AT"].split(":"))
    span = (sh_ * 60 + sm) - (uh * 60 + um)
    if span <= 0:
        span += 24 * 60
    return max(1, -(-span // 60) + 1)


def cmd_up(c, _a) -> int:
    state, bal, raw = status(c)
    if state == "unknown":                  # fail closed: never create blind
        log(c, f"up: REFUSING — cannot determine pod/balance state; not creating a pod.\n{raw}")
        return 1
    if state == "pods":
        log(c, f"up: SKIP — a pod is already running\n{raw}")
        return 0
    if bal is None or bal < float(c["MIN_BALANCE"]):
        log(c, f"up: SKIP — balance ${bal} below MIN_BALANCE ${c['MIN_BALANCE']}")
        return 0
    hours = pod_hours(c)
    log(c, f"up: creating pod for {c['MODEL_LABEL']} (balance ${bal:.2f}, --hours {hours})")
    try:
        r = cloud(c, "up", c["MODEL_LABEL"], "--hours", str(hours))
    except subprocess.TimeoutExpired as e:
        out = e.output if isinstance(e.output, str) else (e.output or b"").decode("utf-8", "replace")
        log(c, "up: TIMED OUT waiting for cloud.sh — A POD MAY BE LIVE AND BILLING. "
               f"Check `gwctl view` / the RunPod console. Partial output:\n{(out or '').strip()[-800:]}")
        return 1
    log(c, (r.stdout or "").strip() or "(no stdout)")
    if r.returncode != 0:
        log(c, f"up: FAILED rc={r.returncode} {(r.stderr or '').strip()[:400]}")
    return r.returncode


def cmd_down(c, _a) -> int:
    """Hardened teardown: kill all, then VERIFY. Only an affirmative "no pods" counts as verified —
    an unreadable status keeps retrying and ends in FAILED, because a half-finished kill keeps billing."""
    for attempt in (1, 2, 3):
        try:
            r = cloud(c, "down", "--all", timeout=600)
            log(c, f"down attempt {attempt}: rc={r.returncode} "
                   f"{(r.stdout or '').strip() or (r.stderr or '').strip()[:300]}")
        except subprocess.TimeoutExpired:
            log(c, f"down attempt {attempt}: TIMED OUT after 600s")
        state, bal, raw = status(c)
        if state == "none":
            log(c, f"down: VERIFIED — no pods alive (balance ${bal:.2f})")
            return 0
        log(c, f"down attempt {attempt}: NOT verified (state={state})\n{raw}")
        if attempt < 3:
            time.sleep(120)
    log(c, "down: FAILED — pods may STILL be running, check RunPod console/billing!")
    return 1


def render_cron(c: dict) -> str:
    uh, um = c["UP_AT"].split(":")
    dh, dm = c["DOWN_AT"].split(":")
    sh_, sm = c["SWEEP_AT"].split(":")
    days = DAY_CRON.get(c["DAYS"], c["DAYS"])
    run = "gwctl"                      # installed entry point — no repo path baked into cron
    return "\n".join([
        BEGIN,
        "# managed by gwctl — edit via `gwctl set`, not by hand",
        f"CRON_TZ={c['TZ']}",
        "PATH=/usr/local/bin:/usr/bin:/bin:" + os.path.expanduser("~/.local/bin"),
        f"{int(um)} {int(uh)} * * {days}\t{run} up",
        f"{int(dm)} {int(dh)} * * *\t{run} down",      # every day: catches one-shot/weekend pods
        f"{int(sm)} {int(sh_)} * * *\t{run} down",     # late sweep: last-resort money guard
        END,
    ])


def read_crontab() -> str:
    """Never turn an unreadable crontab into "" — that would make install/uninstall overwrite the
    user's whole crontab with just our block. Only the benign "no crontab for <user>" is empty."""
    r = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if r.returncode == 0:
        return r.stdout
    if "no crontab for" in (r.stderr or "").lower():
        return ""
    raise RuntimeError(f"cannot read crontab (rc={r.returncode}): {(r.stderr or '').strip()}")


def write_crontab(text: str) -> None:
    subprocess.run(["crontab", "-"], input=text if text.endswith("\n") else text + "\n",
                   text=True, check=True)


def strip_block(text: str) -> str:
    """Remove our managed block, preserving every foreign line. A truncated block (hand-edit or an
    interrupted write) must NOT swallow everything below it — drop only the stray marker."""
    lines = text.splitlines()
    b = next((i for i, l in enumerate(lines) if l.strip() == BEGIN), None)
    if b is None:
        return "\n".join(lines).strip()
    e = next((i for i in range(b + 1, len(lines)) if lines[i].strip() == END), None)
    keep = lines[:b] + (lines[b + 1:] if e is None else lines[e + 1:])
    return "\n".join(keep).strip()


def cmd_install(c, _a) -> int:
    cur = strip_block(read_crontab())
    write_crontab((cur + "\n\n" if cur else "") + render_cron(c))
    log(c, f"install: cron block written (up {c['UP_AT']} {c['DAYS']}, down {c['DOWN_AT']} + sweep "
           f"{c['SWEEP_AT']} daily, TZ {c['TZ']})")
    return 0


def cmd_uninstall(c, _a) -> int:
    write_crontab(strip_block(read_crontab()))
    log(c, "uninstall: cron block removed — NO automatic teardown remains")
    return 0


def cmd_set(c, a) -> int:
    for flag, key in (("up", "UP_AT"), ("down", "DOWN_AT"), ("sweep", "SWEEP_AT"),
                      ("model", "MODEL_LABEL"), ("days", "DAYS"), ("tz", "TZ"),
                      ("min_balance", "MIN_BALANCE"), ("models_dir", "MODELS_DIR"),
                      ("warmup_cmd", "WARMUP_CMD")):
        v = getattr(a, flag, None)
        if v is None:
            continue
        if key.endswith("_AT") and not re.fullmatch(r"([01]?\d|2[0-3]):[0-5]\d", v):
            print(f"bad time for --{flag}: {v} (want HH:MM)", file=sys.stderr)
            return 2
        if key == "DAYS" and v not in DAY_CRON:
            print(f"bad --days: {v} (want {'|'.join(DAY_CRON)})", file=sys.stderr)
            return 2
        if key == "MODELS_DIR" and not Path(os.path.expanduser(v)).is_dir():
            print(f"MODELS_DIR does not exist: '{v}'", file=sys.stderr)
            return 2
        if key == "MIN_BALANCE":            # else float() explodes later, inside cron
            try:
                float(v)
            except ValueError:
                print(f"bad --min-balance: {v} (want a number, e.g. 2.00)", file=sys.stderr)
                return 2
        c[key] = v
    if getattr(a, "model", None) and \
            not (models_dir(c) / c["MODEL_LABEL"] / "serve.env").exists():
        print(f"no serve.env for label '{c['MODEL_LABEL']}' under {models_dir(c)}", file=sys.stderr)
        return 2
    save_conf(c)
    print(f"config saved -> {CONF}")
    if BEGIN in read_crontab():        # keep cron in sync with the config it was generated from
        return cmd_install(c, a)
    print("cron not installed yet — run: gwctl install")
    return 0


def cmd_view(c, _a) -> int:
    print(f"config file : {CONF}{'' if CONF.exists() else '  (defaults — not yet saved)'}")
    for k in DEFAULTS:
        print(f"  {k:<12} = {c[k]}")
    installed = BEGIN in read_crontab()
    print(f"\ncron        : {'INSTALLED' if installed else 'NOT installed'}")
    print("\n".join("  " + l for l in render_cron(c).splitlines()))
    kf = Path(os.path.expanduser(c["KEYS_FILE"]))
    print(f"\nkeys file   : {kf} {'found' if kf.exists() else 'MISSING (or use real env vars)'}")
    state, bal, raw = status(c)
    print(f"\nlive state  : {state}" + (f" | balance ${bal:.2f}" if bal is not None else ""))
    if state == "unknown":
        print("  ⚠ could not read RunPod state (auth/keys/network?) — up refuses and down cannot verify")
    print("\n".join("  " + l for l in raw.splitlines()) or "  (no answer)")
    return 0


def main() -> int:
    from . import metal
    p = argparse.ArgumentParser(description="GPU serve lifecycle: rented pods (up/down) "
                                            "and your own cards (provision/serve/verify)")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("view", "install", "uninstall", "up", "down"):
        sub.add_parser(name)
    s = sub.add_parser("set")
    s.add_argument("--up"); s.add_argument("--down"); s.add_argument("--sweep")
    s.add_argument("--model"); s.add_argument("--days", choices=list(DAY_CRON))
    s.add_argument("--tz"); s.add_argument("--min-balance", dest="min_balance")
    s.add_argument("--models-dir", dest="models_dir")
    s.add_argument("--warmup-cmd", dest="warmup_cmd")
    s = sub.add_parser("provision", help="diagnose (or --apply: fix) driver/docker/toolkit on this box")
    s.add_argument("--apply", action="store_true")
    s = sub.add_parser("render", help="serve.env -> deployment file for a target")
    s.add_argument("label"); s.add_argument("--target", choices=("compose", "k8s"), default="compose")
    s.add_argument("--stdout", action="store_true")
    s = sub.add_parser("serve", help="serve <label> on THIS box: render + up + wait + verify")
    s.add_argument("label")
    s = sub.add_parser("verify", help="health + engine flags + tool-calling acceptance")
    s.add_argument("label", nargs="?"); s.add_argument("--url")
    a = p.parse_args()
    c = load_conf()
    return {"view": cmd_view, "set": cmd_set, "install": cmd_install, "uninstall": cmd_uninstall,
            "up": cmd_up, "down": cmd_down, "provision": metal.cmd_provision,
            "render": metal.cmd_render, "serve": metal.cmd_serve,
            "verify": metal.cmd_verify}[a.cmd](c, a)


if __name__ == "__main__":
    sys.exit(main())
