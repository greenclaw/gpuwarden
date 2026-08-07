# gpuwarden

Rented GPUs bill by the hour, and the expensive failure is never a crash — it's a pod nobody
remembered to stop. `gpuwarden` brings a [RunPod](https://runpod.io) pod up on a schedule, tears it
down **every** evening, and refuses to claim success unless it has *verified* the pod is gone.

The command is `gwctl`; the package is `gpuwarden`.

```bash
gwctl set --up 11:00 --down 18:30 --days mon-fri --model my-model --models-dir ~/models
gwctl install     # schedule it
gwctl view        # config, cron state, live pods, balance
gwctl up          # one-shot, any time — the daily down still reaps it
```

It talks to RunPod's HTTP API and **never touches a local Docker daemon**, so it runs anywhere,
including inside a container without a socket mount.

## Why it exists

Three failure modes, each of which has actually cost money:

- **The pod nobody stopped.** So `down` runs *every* day while `up` runs only on workdays — a manual
  or weekend pod cannot survive the night. No `at`, no one-shot timers, nothing to forget.
- **The teardown that lied.** A status call that fails looks exactly like "no pods running" if you
  squint. `gwctl` never squints: pod state is tri-state (`pods` / `none` / `unknown`), and every
  consumer fails closed. `down` prints VERIFIED only on an affirmative "no pods"; anything else
  retries and then fails loudly. `up` refuses to create anything while state is unknown, because a
  skipped launch costs nothing and a duplicate pod costs money.
- **The endpoint with no password.** Creating a pod without a bearer token would publish an LLM on a
  public URL. `gwctl` refuses to create one, rather than serving it and hoping nobody notices.

## Install

```bash
uv tool install .                    # or: uv tool install "git+https://github.com/<you>/gpuwarden"
uv tool uninstall gpuwarden          # clean removal
```

> Installing from a local path you just edited? Use `--force --reinstall`; `--force` alone can
> silently reuse a cached wheel.

Needs `runpodctl`, `python3`, `bash`, `curl` on the host, plus a cron daemon if you want a schedule.

## Configure

```bash
mkdir -p ~/.config/gpuwarden && chmod 700 ~/.config/gpuwarden
cat > ~/.config/gpuwarden/keys.env <<'EOF'   # you write this; the tool never generates or logs it
RUNPOD_API_KEY=...      # RunPod account API key
VLLM_POD_KEY=...        # bearer the pod endpoint will demand
EOF
chmod 600 ~/.config/gpuwarden/keys.env

mkdir -p ~/models/my-model
cp examples/serve.env.example ~/models/my-model/serve.env   # then edit it
gwctl set --models-dir ~/models --model my-model
```

Config lives in `~/.config/gpuwarden/scheduler.conf` (override with `GPUWARDEN_CONF_DIR`), outside any
checkout, so the tool and your model configs update independently.

| key | meaning |
|---|---|
| `MODELS_DIR` | directory holding `<label>/serve.env` |
| `MODEL_LABEL` | which one to bring up |
| `UP_AT` / `DOWN_AT` / `SWEEP_AT` | schedule; `down` and `sweep` run daily, `up` only on `DAYS` |
| `DAYS` | `mon-fri` or `every` |
| `MIN_BALANCE` | refuse to launch below this, so you never raise a pod that dies at noon |
| `WARMUP_CMD` | optional hook after health; absorbs the cold-start spike |
| `TZ` | timezone the schedule is written in |

## Run it in a container

Cron lives **inside the image**, so the host gains no system daemon and no packages — useful when the
only always-on machine you have is already busy running something else:

```bash
mkdir -p config models          # put keys.env + scheduler.conf in config/, model dirs in models/
echo "TZ=Europe/Moscow" > .env  # REQUIRED: cron fires in container-local time; Debian cron ignores
                                # CRON_TZ, and without this a recreate silently reverts to UTC
docker compose up -d
docker compose logs -f          # the rendered schedule is printed at startup
docker compose down             # gone, nothing left behind
```

The alternative shape — host cron calling `docker run --rm gpuwarden gwctl up` — also works and keeps
the schedule on the host. Pick whichever you'd rather debug at 2am.

## Serve on your own GPU (metal)

The same `serve.env` that drives a rented pod can drive a card you own. The config is the
invariant; the deployment target is just a renderer — so your model definitions survive an
infra move (compose today, Kubernetes tomorrow) unchanged:

```bash
gwctl provision              # diagnose driver/docker/toolkit; --apply executes the fixes
gwctl serve my-model         # render compose + up + wait healthy + verify
gwctl render my-model --target k8s   # reviewed starting-point Deployment+Service
gwctl verify my-model        # or --url http://host:8000 — works against ANY OpenAI endpoint
```

`verify` is the part that keeps you honest: beyond `/health` it checks that the *engine* enabled
what the config asked for (vLLM silently skips prefix caching on hybrid-Mamba models unless told),
and runs a real tool-calling request end to end. `provision` knows the Blackwell trap: the
proprietary kernel module binds to a GB2xx card but cannot initialize it — `nvidia-smi` reports
"No devices were found" while everything else looks healthy; the fix is the `-open` driver.

## Remote control

The scheduler should live on a machine that is always on; a laptop that sleeps will miss its own
teardown. Keep the keys and schedule there, and drive it over SSH — no daemon, no API to secure:

```bash
export GPUWARDEN_HOST=my-server
./remote.sh view
./remote.sh up
```

## Commands

| command | what it does |
|---|---|
| `gwctl view` | config, cron state, rendered schedule, live pods, balance |
| `gwctl set …` | change config; re-renders cron if installed |
| `gwctl install` / `uninstall` | manage a marker-delimited crontab block (foreign lines preserved) |
| `gwctl up` | create a pod — idempotent, balance-guarded, fails closed on unknown state |
| `gwctl down` | terminate everything, then verify; retries; loud failure |

## Known limitation

`runpodctl`'s `--env` accepts only a JSON string, so the endpoint key is briefly visible in that
process's argv during pod creation (readable via `ps` on a shared host). Keep the host single-tenant.
The proper fix is to create pods through RunPod's GraphQL API, which this tool already uses for status
and balance.

## License

MIT — see [LICENSE](LICENSE).
