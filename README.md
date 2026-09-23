# GPUWarden

[![CI](https://github.com/greenclaw/gpuwarden/actions/workflows/ci.yml/badge.svg)](https://github.com/greenclaw/gpuwarden/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Rented GPUs bill by the hour, and the expensive failure is never a crash — it's a pod nobody
remembered to stop. GPUWarden brings a [RunPod](https://runpod.io) pod up on a schedule, tears it
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

## How it compares

The neighborhood splits into two camps, and GPUWarden sits in the gap between them (survey
2026-08; stars/activity change, the shapes don't):

| | schedule semantics | teardown | serving | shape |
|---|---|---|---|---|
| [runpod-auto-stop](https://github.com/stlaurentjr/runpod-auto-stop), [runpod-budget](https://github.com/gitlost-murali/runpod-budget) | idle/runtime **stop only** | fire and forget | — | single script |
| [Runpod-Idle-Pod-Monitor](https://github.com/runpod/Runpod-Idle-Pod-Monitor) (official) | reactive idle monitor | fire and forget | — | monitor pod + web UI |
| [SkyPilot](https://github.com/skypilot-org/skypilot) / [dstack](https://github.com/dstackai/dstack) | idle-based autostop/autodown | not verified | recipes / services, replicas, autoscaling | multi-cloud platform with a controller |
| **GPUWarden** | **workday clock**: `up` weekday mornings, `down` *every* evening | **verified, fail-closed** (tri-state, retries, loud failure) | `serve.env` invariant → compose/k8s renderers + engine-flag & tool-calling `verify` | one cron block + a CLI |

What the others don't do: nobody models a *workday* (every tool in the small camp waits for
idleness; a pod idling at 2am has already cost you the night), and nobody verifies that a
teardown actually happened — they call the stop API and believe it. Those two are the reasons
this tool exists.

What the platforms do better: SkyPilot shops across clouds for the cheapest GPU, does replicas,
autoscaling and spot; dstack adds a full control plane. If you need fleet orchestration, use
them — GPUWarden is deliberately the small tool for the "one team, one model, rented by the
hour" shape, where the platform's controller VM would cost more than the problem.

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

Two environment switches make `up` fail closed instead of leaving a pod billing: `GW_ON_HEALTH_TIMEOUT=terminate`
tears the pod down when it never becomes healthy, and `GW_WARMUP_STRICT=1` tears it down when `WARMUP_CMD` fails
three times (the hook retries either way). Unset, both keep the pod and warn.

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

`provision` also catches the failure that bites a box left alone for weeks: unattended upgrades replace
the driver's userspace under a running serve, nothing breaks until the next container restart, and then
it breaks three ways at once (NVML version mismatch, a CDI spec pointing at deleted libraries, a stopped
`nvidia-persistenced`). It reports module/userspace drift and stale CDI paths, and checks that the driver
packages are held (`--apply` holds them).

### Changing a live serve

```bash
gwctl serve my-model --replace     # take the GPU from whatever serves on it now
gwctl serve my-model --recreate    # same config, fresh engine (empty prefix cache)
```

`--replace` is built for production changes (an image bump, a flag, taking over a hand-made
container):

- a container that already holds the target name is **moved aside** (`<name>-replaced-<stamp>`), not
  deleted; other serves on the card are stopped;
- the new serve must come up **and** pass `verify`; an engine that dies at startup fails in seconds,
  not after the health timeout;
- on failure the new container is removed, the moved-aside one gets its name back, and whatever was
  serving is started again; on success the moved-aside container is removed.

Set `CONTAINER_NAME` in `serve.env` to keep an existing container's name (log shippers and dashboards
often key on it); otherwise gwctl names it `gw-<label>`. An upgrade is then a one-line diff to `IMAGE`
plus `gwctl serve <label> --replace`, and the rollback is the same command after reverting the line.

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
| `gwctl provision [--apply]` | metal: driver/module drift, stale CDI, driver hold, docker + toolkit |
| `gwctl serve <label> [--replace] [--recreate]` | metal: render compose, start, wait healthy, verify; rolls back on failure |
| `gwctl verify <label> [--url …] [--container …]` | health, engine facts from the startup log, tool-calling acceptance |
| `gwctl render <label> [--target k8s] [--stdout]` | write the compose (or a starting-point Deployment) without running it |

## Known limitation

`runpodctl`'s `--env` accepts only a JSON string, so the endpoint key is briefly visible in that
process's argv during pod creation (readable via `ps` on a shared host). Keep the host single-tenant.
The proper fix is to create pods through RunPod's GraphQL API, which this tool already uses for status
and balance.

## License

MIT — see [LICENSE](LICENSE).
