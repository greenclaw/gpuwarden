"""cloud.sh pod creation: what reaches `runpodctl pod create`, with a fake runpodctl on PATH."""
import os
import pathlib
import subprocess

CLOUD = pathlib.Path(__file__).resolve().parents[1] / "src/gpuwarden/cloud.sh"
SERVE_ENV = """IMAGE=vllm/vllm-openai:v0@sha256:00
MODEL=org/m
SERVED_NAME=m
TOOL_PARSER=hermes
MAXLEN=4096
"""


def create_argv(tmp_path, extra_env=""):
    (tmp_path / "bin").mkdir()
    fake = tmp_path / "bin/runpodctl"
    fake.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > {tmp_path}/argv\n')
    fake.chmod(0o755)
    (tmp_path / "models/lbl").mkdir(parents=True)
    (tmp_path / "models/lbl/serve.env").write_text(SERVE_ENV + extra_env)
    env = dict(os.environ, PATH=f"{tmp_path}/bin:{os.environ['PATH']}", MODELS_DIR=f"{tmp_path}/models",
               RUNPOD_API_KEY="k", VLLM_POD_KEY="p")
    subprocess.run(["bash", str(CLOUD), "up", "lbl"], env=env, capture_output=True, timeout=30)
    return (tmp_path / "argv").read_text().split("\n")


def test_min_cuda_reaches_pod_create(tmp_path):
    argv = create_argv(tmp_path, "MIN_CUDA=13.0\n")
    assert argv[argv.index("--min-cuda-version") + 1] == "13.0"


def test_no_min_cuda_no_flag(tmp_path):
    assert "--min-cuda-version" not in create_argv(tmp_path)


def fake_bin(tmp_path, name, body):
    (tmp_path / "bin").mkdir(exist_ok=True)
    f = tmp_path / "bin" / name
    f.write_text("#!/usr/bin/env bash\n" + body)
    f.chmod(0o755)


def test_create_passes_no_flag_removed_from_runpodctl_v3(tmp_path):
    # REST v1 (rest.runpod.io) was retired 2026-11-15; runpodctl >= 2.15 has no --terminate-after and v2 has
    # no server-side auto-terminate, so passing it fails every create with "unknown flag".
    assert "--terminate-after" not in create_argv(tmp_path)


def test_status_reads_uptime_from_runpodctl_v3_list(tmp_path):
    # v3 `pod list` shape: uptimeSeconds at the top level
    fake_bin(tmp_path, "runpodctl",
             """echo '[{"id":"p1","name":"gw-x","desiredStatus":"RUNNING","costPerHr":2,"uptimeSeconds":5400}]'\n""")
    fake_bin(tmp_path, "curl", """echo '{"data":{"myself":{"clientBalance":10,"currentSpendPerHr":2}}}'\n""")
    env = dict(os.environ, PATH=f"{tmp_path}/bin:{os.environ['PATH']}", RUNPOD_API_KEY="k")
    r = subprocess.run(["bash", str(CLOUD), "status"], env=env, capture_output=True, text=True, timeout=30)
    assert "up 1h30m | spent ~$3.00" in r.stdout
    assert "balance $10.00" in r.stdout


def test_stderr_noise_does_not_hide_a_created_pod(tmp_path):
    # runpodctl v3 may print an update notice / config warning on stderr before the success JSON; merged
    # into stdout it broke the id parse and a billing pod was reported as "URGENT ... could not be parsed".
    fake_bin(tmp_path, "runpodctl", 'echo "runpodctl v9 is available" >&2\necho \'{"id":"p1"}\'\n')
    fake_bin(tmp_path, "curl", "printf 200\n")   # the pod's /health answers 200 on the first poll
    (tmp_path / "models/lbl").mkdir(parents=True)
    (tmp_path / "models/lbl/serve.env").write_text(SERVE_ENV)
    env = dict(os.environ, PATH=f"{tmp_path}/bin:{os.environ['PATH']}", MODELS_DIR=f"{tmp_path}/models",
               RUNPOD_API_KEY="k", VLLM_POD_KEY="test-endpoint-key-7f3a")
    r = subprocess.run(["bash", str(CLOUD), "up", "lbl"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "URGENT" not in r.stderr
    assert "READY: https://p1-8000.proxy.runpod.net/v1" in r.stdout


def test_create_failure_reason_is_reported(tmp_path):
    # under `set -e` + pipefail a failed runpodctl killed cloud.sh at the assignment, before the reason was printed
    fake_bin(tmp_path, "runpodctl", 'echo \'{"error":"no instances available"}\' >&2\nexit 1\n')
    (tmp_path / "models/lbl").mkdir(parents=True)
    (tmp_path / "models/lbl/serve.env").write_text(SERVE_ENV)
    env = dict(os.environ, PATH=f"{tmp_path}/bin:{os.environ['PATH']}", MODELS_DIR=f"{tmp_path}/models",
               RUNPOD_API_KEY="k", VLLM_POD_KEY="test-endpoint-key-7f3a")
    r = subprocess.run(["bash", str(CLOUD), "up", "lbl"], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 1
    assert "pod create failed" in r.stderr and "no instances available" in r.stderr
