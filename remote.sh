#!/usr/bin/env bash
# Local thin client — runs gwctl ON the dev server, which is the single source of truth
# (schedule + config + RunPod keys live there; this laptop holds none). SSH *is* the RPC: no daemon.
# The server has gwctl installed (`uv tool install`), so no repo path is needed here.
#
#   export GPUWARDEN_HOST=dev-server     # ssh host/alias, ideally from ~/.ssh/config
#
#   vllm/gpuwarden/remote.sh view
#   vllm/gpuwarden/remote.sh up              # one-shot: the server's daily down+sweep tear it down
#   vllm/gpuwarden/remote.sh down
#   vllm/gpuwarden/remote.sh set --up 11:00 --down 18:30
set -euo pipefail
HOST="${GPUWARDEN_HOST:?set GPUWARDEN_HOST to the dev server ssh host/alias}"
[ $# -gt 0 ] || { sed -n '2,11p' "$0"; exit 1; }
# $* would be re-parsed by the remote shell: an arg with a space/quote mangles, and anything
# non-interactive calling this would be an injection vector. %q-quote each arg instead.
exec ssh "$HOST" "$(printf '%q ' gwctl "$@")"
