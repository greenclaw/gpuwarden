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
