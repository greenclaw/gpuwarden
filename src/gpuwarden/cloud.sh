#!/usr/bin/env bash
# shellcheck disable=SC1090  # sources the user's serve.env, path known only at runtime
# cloud.sh — RunPod backend for gpuwarden: serve a model from its serve.env on a rented GPU pod.
#
#   cloud.sh up <label> [--gpu "NVIDIA RTX PRO 5000 Blackwell"] [--hours 6] [--cloud COMMUNITY]
#   cloud.sh status                 # pods + uptime + $/h + spent + account balance
#   cloud.sh down <podId|--all>     # terminate + report final cost
#
# Normally invoked by `gwctl`, which supplies the environment. Standalone use needs:
#   MODELS_DIR      dir containing <label>/serve.env          (required)
#   RUNPOD_API_KEY  RunPod account API key                    (required)
#   VLLM_POD_KEY    bearer the pod endpoint will demand       (required, must be non-empty)
#   WARMUP_CMD      optional post-health hook; receives BASE_URL / SERVED_MODEL / LABEL in its env
#
# Talks to RunPod's HTTP API only — it never touches a local Docker daemon, so it is safe to run
# inside a container without a socket mount.
set -euo pipefail

: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY (gwctl loads it from its keys file)}"

gql() {  # gql '<query>' — key via stdin header, never argv
  printf "Authorization: Bearer %s\nContent-Type: application/json\n" "$RUNPOD_API_KEY" |
    curl -s -m 20 -H @- -X POST https://api.runpod.io/graphql -d "{\"query\":\"$1\"}"
}

balance() {
  gql '{ myself { clientBalance currentSpendPerHr } }' | python3 -c '
import sys, json
d = json.load(sys.stdin)["data"]["myself"]
print("balance $%.2f | spend $%s/h" % (d["clientBalance"], d["currentSpendPerHr"]))'
}

case "${1:-}" in
  up)
    LABEL="${2:?usage: cloud.sh up <label>}"; shift 2
    GPU="NVIDIA RTX PRO 5000 Blackwell"; HOURS=6; CLOUD="COMMUNITY"
    while [ $# -gt 0 ]; do case "$1" in
      --gpu) GPU="$2"; shift 2;; --hours) HOURS="$2"; shift 2;; --cloud) CLOUD="$2"; shift 2;;
      *) echo "unknown flag $1" >&2; exit 1;; esac; done
    : "${MODELS_DIR:?set MODELS_DIR to the directory holding <label>/serve.env}"
    ENVF="$MODELS_DIR/$LABEL/serve.env"
    [ -f "$ENVF" ] || { echo "no $ENVF — see examples/serve.env.example" >&2; exit 1; }
    set -a; . "$ENVF"; set +a

    REASON_ARG=""; [ -n "${REASONING_PARSER:-}" ] && [ "$REASONING_PARSER" != "none" ] && REASON_ARG="--reasoning-parser $REASONING_PARSER"
    # Pin the weights to a commit when serve.env carries REVISION, else HF 'main' floats.
    REV_ARG=""; [ -n "${REVISION:-}" ] && REV_ARG="--revision $REVISION"
    ARGS="--model ${MODEL:?serve.env must set MODEL} $REV_ARG \
--served-model-name ${SERVED_NAME:?serve.env must set SERVED_NAME} --trust-remote-code \
--enable-auto-tool-choice --tool-call-parser ${TOOL_PARSER:?serve.env must set TOOL_PARSER} $REASON_ARG \
--enable-prefix-caching --max-model-len ${MAXLEN:?serve.env must set MAXLEN} ${EXTRA_ARGS:-}"

    # Fail closed: `set -u` catches UNSET but not EMPTY. An empty key would serve the model on a
    # public proxy URL with NO bearer — a world-reachable LLM endpoint on your account.
    [ -n "${VLLM_POD_KEY:-}" ] || {
      echo "[cloud] FATAL: VLLM_POD_KEY is empty — refusing to create an unauthenticated public endpoint" >&2
      exit 1; }
    # Redact via a literal-match pattern FILE: keeps the key out of grep's argv (visible in ps on a
    # shared host) and out of BRE interpretation (a key containing [ or * would mis-match or error).
    PATFILE=$(mktemp); chmod 600 "$PATFILE"; trap 'rm -f "$PATFILE"' EXIT
    printf '%s\n' "$VLLM_POD_KEY" > "$PATFILE"

    TERM_AT=$(date -u -v+"${HOURS}"H +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d "+${HOURS} hours" +%Y-%m-%dT%H:%M:%SZ)
    echo "[cloud] creating pod: $LABEL on '$GPU' ($CLOUD), auto-terminate $TERM_AT"
    OUT=$(runpodctl pod create --name "gw-$LABEL" --gpu-id "$GPU" --cloud-type "$CLOUD" \
      --image "${IMAGE:?serve.env must set IMAGE (pin it by digest)}" \
      --container-disk-in-gb "${DISK_GB:-80}" \
      --ports "8000/http" --terminate-after "$TERM_AT" \
      --env "{\"VLLM_API_KEY\":\"$VLLM_POD_KEY\"}" --docker-args "$ARGS" 2>&1 | { grep -vF -f "$PATFILE" || true; })
    # KNOWN LIMITATION: runpodctl's --env takes only a JSON string, so the endpoint key is unavoidably
    # in runpodctl's argv for the duration of the create call (readable via ps on a shared host).
    # Keep hosts single-tenant; the proper fix is to create pods through the GraphQL API (see gql()).

    PID=$(printf '%s' "$OUT" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("id",""))
except Exception: print("")' || true)
    if [ -z "$PID" ]; then
      # Distinguish "create failed" from "created but I cannot parse the id" — the latter means a pod
      # may be BILLING with no id recorded anywhere, which must never look like a plain failure.
      if printf '%s' "$OUT" | grep -qiE '"id"|created|pod .* started'; then
        echo "[cloud] URGENT: pod may have been CREATED but its id could not be parsed." >&2
        echo "[cloud] Check the RunPod console NOW and terminate it manually. Raw output:" >&2
      else
        echo "[cloud] pod create failed:" >&2
      fi
      printf '%s\n' "$OUT" | head -10 >&2
      exit 1
    fi

    BASE="https://${PID}-8000.proxy.runpod.net"
    echo "[cloud] pod $PID | $BASE/v1"
    echo "[cloud] waiting for health (image pull + weights + compile: 10-25 min on community)..."
    for _ in $(seq 1 120); do
      code=$(curl -s -m 8 -o /dev/null -w "%{http_code}" "$BASE/health" || true)
      [ "$code" = "200" ] && {
        if [ -n "${WARMUP_CMD:-}" ]; then
          echo "[cloud] HEALTHY — running warmup hook..."
          BASE_URL="$BASE/v1" SERVED_MODEL="$SERVED_NAME" LABEL="$LABEL" \
            VLLM_API_KEY="$VLLM_POD_KEY" bash -c "$WARMUP_CMD" \
            || echo "[cloud] WARNING: warmup hook failed — first request will carry the cold-start spike"
        else
          echo "[cloud] HEALTHY (no WARMUP_CMD set — the first request will be slow)"
        fi
        echo "[cloud] READY: $BASE/v1"
        exit 0
      }
      sleep 20
    done
    echo "[cloud] health timeout after ~40min — THE POD IS STILL RUNNING AND BILLING." >&2
    echo "[cloud] terminate it with: cloud.sh down $PID" >&2
    exit 2
    ;;
  status)
    runpodctl pod list -o json 2>/dev/null | python3 -c '
import sys, json
pods = json.load(sys.stdin)
pods = pods if isinstance(pods, list) else pods.get("pods", [])
for p in pods:
    up = (p.get("runtime") or {}).get("uptimeInSeconds") or 0
    cost = p.get("costPerHr") or 0
    print(p.get("id"), "|", p.get("name"), "|", p.get("desiredStatus"),
          "| $%s/h" % cost, "| up %dh%02dm" % (up // 3600, up % 3600 // 60),
          "| spent ~$%.2f" % (up / 3600 * cost))
if not pods:
    print("no pods")'
    balance
    ;;
  down)
    TARGET="${2:?usage: cloud.sh down <podId|--all>}"
    rc=0
    if [ "$TARGET" = "--all" ]; then
      ids=$(runpodctl pod list -o json | python3 -c '
import sys, json
pods = json.load(sys.stdin)
pods = pods if isinstance(pods, list) else pods.get("pods", [])
[print(p["id"]) for p in pods]')
      # Attempt EVERY pod: one stuck delete must not leave the rest billing (`set -e` would abort).
      for id in $ids; do
        if runpodctl pod delete "$id"; then echo "[cloud] terminated $id"
        else echo "[cloud] FAILED to terminate $id" >&2; rc=1; fi
      done
    else
      runpodctl pod delete "$TARGET" && echo "[cloud] terminated $TARGET" || { echo "[cloud] FAILED to terminate $TARGET" >&2; rc=1; }
    fi
    balance
    exit $rc
    ;;
  *) sed -n '2,13p' "$0"; exit 1;;
esac
