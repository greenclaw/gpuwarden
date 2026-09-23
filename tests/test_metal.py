"""Pure-function checks behind `gwctl provision / serve / verify`.

Each case is a state a real box was caught in; the parsers must name it, not guess."""
from gpuwarden.metal import (
    adopt_rename,
    compose_project,
    container_crashed,
    container_name,
    driver_drift,
    driver_packages,
    engine_facts,
    module_version,
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


def test_same_name_from_another_compose_project_is_renamed_out_of_the_way():
    # /opt/vllm made "vllm-prod"; gwctl's project must take the name but keep the old one restorable
    assert adopt_rename("vllm-prod", "vllm", "mylabel", "20260923T1200") == "vllm-prod-replaced-20260923T1200"


def test_our_own_container_or_no_container_needs_no_rename():
    assert adopt_rename("vllm-prod", "mylabel", "mylabel", "t") is None
    assert adopt_rename("vllm-prod", None, "mylabel", "t") is None


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
