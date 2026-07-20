#!/usr/bin/env bash
# Writes the schedule into the container's own crontab, then runs cron in the foreground.
# The host's crontab is never touched — that is the whole point of the container shape.
set -euo pipefail

: "${GPUWARDEN_CONF_DIR:=/config}"
export GPUWARDEN_CONF_DIR

if [ ! -f "$GPUWARDEN_CONF_DIR/scheduler.conf" ]; then
  echo "no $GPUWARDEN_CONF_DIR/scheduler.conf — run 'gwctl set …' against the mounted config dir first" >&2
  exit 1
fi
if [ ! -f "$GPUWARDEN_CONF_DIR/keys.env" ] && [ -z "${RUNPOD_API_KEY:-}" ]; then
  echo "no keys: mount $GPUWARDEN_CONF_DIR/keys.env or pass RUNPOD_API_KEY/VLLM_POD_KEY in the env" >&2
  exit 1
fi

gwctl install                      # renders the marker block into root's crontab in THIS container
gwctl view || true                 # one-shot visibility in `docker logs`

# cron does not inherit the container env, so hand the keys file location to the jobs.
printenv | grep -E '^(GPUWARDEN_|RUNPOD_API_KEY|VLLM_POD_KEY)' > /etc/environment || true

echo "[gpuwarden] cron running in foreground; schedule above"
exec cron -f -L 2
