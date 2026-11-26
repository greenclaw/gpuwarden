"""Pure-function checks behind `gwctl provision / serve / verify`.

Each case is a state a real box was caught in; the parsers must name it, not guess."""
import json
import os
import shutil
import subprocess

import pytest

from gpuwarden.metal import (
    aside_action,
    compose_project,
    container_crashed,
    container_name,
    driver_drift,
    driver_packages,
    engine_facts,
    extra_api_keys,
    module_version,
    render_compose,
    rollback_plan,
    serve_claimants,
    stale_cdi_paths,
)

PROC_OLD = ("NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  595.71.05  Release Build  "
            "(dvs-builder@U22-I3-G08-03-1)  Fri Apr 24 06:42:30 UTC 2026\n")


def test_module_version_parses_open_module():
    assert module_version(PROC_OLD) == "595.71.05"


def test_module_version_absent_when_no_module_loaded():
    assert module_version("") is None


def test_drift_detected_when_userspace_upgraded_under_loaded_module():
    # unattended-upgrades bumped userspace while a serve held the old module loaded
    note = driver_drift("595.71.05", ["/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.595.91.07"])
    assert note and "595.71.05" in note and "595.91.07" in note


def test_no_drift_when_versions_match():
    assert driver_drift("595.91.07", ["/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.595.91.07"]) is None


def test_no_drift_verdict_without_evidence():
    assert driver_drift(None, ["/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.595.91.07"]) is None
    assert driver_drift("595.91.07", []) is None


CDI = """
containerEdits:
  mounts:
  - hostPath: /run/nvidia-persistenced/socket
    containerPath: /run/nvidia-persistenced/socket
  - hostPath: /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.595.71.05
    containerPath: /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.595.71.05
  - hostPath: /usr/lib/x86_64-linux-gnu/libcuda.so.595.91.07
    containerPath: /usr/lib/x86_64-linux-gnu/libcuda.so.595.91.07
"""


def test_stale_cdi_lists_every_missing_mount_source():
    present = {"/usr/lib/x86_64-linux-gnu/libcuda.so.595.91.07"}
    missing = stale_cdi_paths(CDI, exists=lambda p: p in present)
    assert missing == ["/run/nvidia-persistenced/socket",
                       "/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.595.71.05"]


def test_fresh_cdi_has_no_missing_paths():
    assert stale_cdi_paths(CDI, exists=lambda p: True) == []


def test_claimants_are_other_vllm_serves_on_the_gpu():
    running = [
        {"name": "vllm-prod", "image": "vllm/vllm-openai:v0.24.0@sha256:x", "gpu": True},
        {"name": "dcgm-exporter", "image": "nvidia/dcgm-exporter:4.6.0", "gpu": True},
        {"name": "gw-mymodel", "image": "vllm/vllm-openai:v0.30.0@sha256:y", "gpu": True},
        {"name": "vmagent", "image": "victoriametrics/vmagent:v1", "gpu": False},
    ]
    # monitoring may share the card; another serve may not; our own container is not a rival
    assert serve_claimants(running, own="gw-mymodel") == ["vllm-prod"]


LOG = """
(APIServer pid=1) WARNING [config.py:422] Mamba cache mode is set to 'align' for X by default
(EngineCore pid=230) INFO [core.py:114] Initializing a V1 LLM engine (v0.24.0) with config: model='m', kv_cache_dtype=auto, enable_prefix_caching=True
(EngineCore pid=230) INFO [nvfp4.py:270] Using 'MARLIN' NvFp4 MoE backend out of potential backends: ['FLASHINFER_TRTLLM', 'MARLIN'].
(EngineCore pid=230) WARNING [marlin.py:34] Your GPU does not have native support for FP4 computation but FP4 quantization is being used. Weight-only FP4 compression will be used leveraging the Marlin kernel.
(EngineCore pid=230) INFO [interface.py:773] Setting attention block size to 1056 tokens to ensure that attention page size is >= mamba page size.
(EngineCore pid=230) INFO [gpu_model_runner.py:5255] Model loading took 20.4 GiB memory and 12.157034 seconds
(EngineCore pid=230) INFO [gpu_worker.py:508] Available KV cache memory: 64.16 GiB
(EngineCore pid=230) INFO [kv_cache_utils.py:2146] GPU KV cache size: 3,274,229 tokens
(EngineCore pid=230) INFO [kv_cache_utils.py:2147] Maximum concurrency for 262,144 tokens per request: 12.49x
"""


def test_engine_facts_from_startup_log():
    f = engine_facts(LOG)
    assert f["vllm_version"] == "0.24.0"
    assert f["kv_cache_tokens"] == 3274229
    assert f["max_concurrency"] == "12.49x @ 262,144"
    assert f["kv_cache_dtype"] == "auto"
    assert f["attention_block_size"] == 1056
    assert f["mamba_cache_mode"] == "align"
    assert f["nvfp4_moe_backend"] == "MARLIN"
    assert f["fp4_native"] is False       # the fallback vLLM itself warns about
    assert f["weights_gib"] == 20.4       # what a vision-encoder-off start would shrink
    assert f["kv_cache_gib"] == 64.16


def test_engine_facts_empty_log_gives_empty_facts():
    assert engine_facts("") == {}


def test_crash_detected_when_container_exited():
    # a new vLLM version dying at startup must fail serve now, not after a 60-minute health wait
    assert container_crashed("exited 0") == "container exited"


def test_crash_detected_when_restart_policy_loops_it():
    # restart: unless-stopped turns a startup crash into a quiet restart loop
    assert container_crashed("running 3") == "container restarted 3x during startup"
    assert container_crashed("restarting 1") == "container restarted 1x during startup"


def test_healthy_start_is_not_a_crash():
    assert container_crashed("running 0") is None
    assert container_crashed("") is None          # inspect raced the create — keep waiting


def test_rollback_stops_the_failed_serve_and_restarts_what_it_replaced():
    # a failed --replace must not leave the card empty: the new serve goes, the old ones come back
    assert rollback_plan("gw-new", ["vllm-prod"]) == [
        ["docker", "stop", "gw-new"], ["docker", "start", "vllm-prod"]]


def test_rollback_without_replace_still_ends_the_restart_loop():
    assert rollback_plan("gw-new", []) == [["docker", "stop", "gw-new"]]


DPKG = """ii |libnvidia-compute-595-server
hi |nvidia-driver-595-server-open
un |nvidia-driver-595-server
ii |nvidia-container-toolkit
rc |nvidia-dkms-595-server
"""


def test_driver_packages_counts_held_ones_as_installed():
    # a held package reports 'hi'; dropping it made the check vanish right after --apply held it
    assert driver_packages(DPKG) == ["libnvidia-compute-595-server", "nvidia-driver-595-server-open"]


def test_container_name_defaults_to_gw_label_and_honours_override():
    assert container_name({"_label": "m"}) == "gw-m"
    # an existing serve keeps its name when it moves under gwctl (log labels, dashboards)
    assert container_name({"_label": "m", "CONTAINER_NAME": "vllm-prod"}) == "vllm-prod"


def test_compose_project_is_the_normalised_label_directory():
    assert compose_project("Qwen36-35B.ccappai") == "qwen36-35bccappai"


def test_rollback_gives_an_adopted_container_its_name_back():
    plan = rollback_plan("vllm-prod", ["vllm-prod-replaced-t"], {"vllm-prod-replaced-t": "vllm-prod"})
    assert plan == [["docker", "stop", "vllm-prod"], ["docker", "rm", "vllm-prod"],
                    ["docker", "rename", "vllm-prod-replaced-t", "vllm-prod"],
                    ["docker", "start", "vllm-prod"]]


def test_rollback_renames_back_an_adopted_container_that_was_not_running():
    # the adopted container was stopped: it gets its name back but is not started
    plan = rollback_plan("vllm-prod", ["gw-other"], {"vllm-prod-replaced-t": "vllm-prod"})
    assert plan == [["docker", "stop", "vllm-prod"], ["docker", "rm", "vllm-prod"],
                    ["docker", "rename", "vllm-prod-replaced-t", "vllm-prod"],
                    ["docker", "start", "gw-other"]]


def test_replace_moves_our_own_container_aside_so_an_upgrade_can_roll_back():
    # an image bump recreates the container in place — the old one would be gone for a rollback
    assert aside_action(exists_project="lbl", project="lbl", replace=True) == "aside"
    assert aside_action(exists_project="vllm", project="lbl", replace=True) == "aside"


def test_plain_serve_stays_idempotent_and_refuses_foreign_names():
    assert aside_action(exists_project="lbl", project="lbl", replace=False) == "inplace"
    assert aside_action(exists_project="vllm", project="lbl", replace=False) == "refuse"
    assert aside_action(exists_project=None, project="lbl", replace=True) == "fresh"


E = {"_label": "m", "IMAGE": "img@sha256:0", "MODEL": "org/m", "SERVED_NAME": "m",
     "TOOL_PARSER": "some-parser", "MAXLEN": "4096"}


def test_compose_takes_keys_through_api_key_interpolation_never_values():
    out = render_compose(E)
    assert "--api-key ${VLLM_API_KEY:?" in out and "${VLLM_EXTRA_API_KEYS:-}" in out
    assert "--config" not in out and "api-keys" not in out
    assert "fake-" not in out                           # values only ever come from the environment


@pytest.mark.skipif(not shutil.which("docker"), reason="needs the docker compose CLI (config only, no daemon)")
def test_compose_argv_one_entry_per_key_and_a_key_change_changes_the_command(tmp_path):
    f = tmp_path / "compose.yaml"
    f.write_text(render_compose(E))

    def argv(extras):
        r = subprocess.run(["docker", "compose", "-f", str(f), "config", "--format", "json"],
                           env={**os.environ, "VLLM_API_KEY": "fake-main", "VLLM_EXTRA_API_KEYS": extras},
                           capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip("docker compose config unavailable")
        return json.loads(r.stdout)["services"]["vllm"]["command"]

    none, two, other = argv(""), argv("fake-a fake-b"), argv("fake-a fake-c")
    assert none[-2:] == ["--api-key", "fake-main"]                  # empty extras: no empty argv entry
    assert two[-4:] == ["--api-key", "fake-main", "fake-a", "fake-b"]
    assert two != other and two != none      # different command => `up -d` recreates the container


def test_extra_keys_split_on_commas_and_whitespace():
    assert extra_api_keys({"VLLM_EXTRA_API_KEYS": "fake-a, fake-b\nfake-c fake-d"}, "fake-main") == \
        ["fake-a", "fake-b", "fake-c", "fake-d"]


@pytest.mark.parametrize("raw", [None, "", "  \n "])
def test_no_extras_is_fine(raw):
    assert extra_api_keys({"VLLM_EXTRA_API_KEYS": raw} if raw is not None else {}, "fake-main") == []


@pytest.mark.parametrize("raw,why", [
    ("fake-a,,fake-b", "empty"), ("fake-a,", "empty"), ("fake-a fake-a", "duplicate"),
    ("fake-main", "main key"), ("fa'ke", "unsafe"), ('fa"ke', "unsafe"), ("fa$ke", "unsafe"),
    ("fa\\ke", "unsafe"), ("-fakeX", "unsafe"), ("fake;x", "unsafe"), ("fake&x", "unsafe")])
def test_bad_extras_refuse_without_printing_a_key(raw, why):
    with pytest.raises(SystemExit) as ex:
        extra_api_keys({"VLLM_EXTRA_API_KEYS": raw}, "fake-main")
    msg = str(ex.value)
    assert msg.startswith("FATAL") and why in msg and "fake" not in msg and raw not in msg


@pytest.mark.parametrize("main", ["fa ke", "-fake-main", "fake;main", ""])
def test_unsafe_main_key_refuses_without_printing_it(main):
    with pytest.raises(SystemExit) as ex:
        extra_api_keys({}, main)
    assert "main API key" in str(ex.value) and "VLLM_API_KEY" in str(ex.value)   # source named generically
    assert "fake" not in str(ex.value)


def _serve_fixture(tmp_path, monkeypatch, up_rc):
    import argparse

    from gpuwarden import metal
    (tmp_path / "models" / "m").mkdir(parents=True)
    (tmp_path / "models" / "m" / "serve.env").write_text(
        "IMAGE=img@sha256:0\nMODEL=org/m\nSERVED_NAME=m\nTOOL_PARSER=some-parser\nMAXLEN=4096\n")
    kf = tmp_path / "cfg" / "keys.env"
    c = {"MODELS_DIR": str(tmp_path / "models"), "KEYS_FILE": str(kf), "LOG_FILE": str(tmp_path / "log")}
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw.get("env")))
        rc = up_rc if cmd[:2] == ["docker", "compose"] else 1 if cmd[:2] == ["docker", "inspect"] else 0
        return subprocess.CompletedProcess(cmd, rc, "", "")

    monkeypatch.setattr(metal.subprocess, "run", fake_run)
    monkeypatch.setattr(metal, "running_containers", lambda: [])
    monkeypatch.setenv("VLLM_API_KEY", "fake-main")
    monkeypatch.setenv("VLLM_EXTRA_API_KEYS", "fake-qa")
    return metal, c, kf, calls, argparse.Namespace(label="m", replace=False)


def test_serve_hands_keys_to_compose_through_the_up_env_and_writes_no_key_file(tmp_path, monkeypatch, capsys):
    metal, c, kf, calls, a = _serve_fixture(tmp_path, monkeypatch, up_rc=0)
    monkeypatch.setattr(metal, "_get", lambda *x, **y: (200, ""))
    monkeypatch.setattr(metal, "_verify", lambda *x, **y: 0)
    assert metal.cmd_serve(c, a) == 0
    (_, env), = [x for x in calls if x[0][:2] == ["docker", "compose"]]
    assert env["VLLM_API_KEY"] == "fake-main" and env["VLLM_EXTRA_API_KEYS"] == "fake-qa"
    assert not kf.parent.exists()                       # nothing wrote a key file
    out = capsys.readouterr().out + (tmp_path / "log").read_text()
    assert "1 extra API key(s)" in out and "fake-" not in out
    assert "fake-" not in (tmp_path / "models" / "m" / "compose.yaml").read_text()


def test_bad_key_refuses_before_anything_is_stopped(tmp_path, monkeypatch):
    metal, c, kf, calls, a = _serve_fixture(tmp_path, monkeypatch, up_rc=0)
    monkeypatch.setenv("VLLM_EXTRA_API_KEYS", "-fake-qa")
    with pytest.raises(SystemExit):
        metal.cmd_serve(c, a)
    assert calls == []
