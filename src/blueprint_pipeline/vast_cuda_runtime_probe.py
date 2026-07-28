"""Container-local CUDA runtime admission for Vast paid GPU bundles."""

from __future__ import annotations

import re
from typing import Any, Mapping


def cuda_runtime_probe_shell_fragment(*, required: bool) -> str:
    if not required:
        return "cuda_runtime_rc=0; echo BLUEPRINT_VAST_CUDA_RUNTIME_SKIPPED; "
    return (
        'CUDA_PY="${PY_NET:-}"; cuda_runtime_rc=1; '
        'if [ -z "$CUDA_PY" ] && [ -x /opt/conda/bin/python ]; then CUDA_PY=/opt/conda/bin/python; '
        'elif [ -z "$CUDA_PY" ] && [ -x /usr/local/bin/python ]; then CUDA_PY=/usr/local/bin/python; '
        'elif [ -z "$CUDA_PY" ] && [ -x /isaac-sim/python.sh ]; then CUDA_PY=/isaac-sim/python.sh; '
        'elif [ -z "$CUDA_PY" ] && [ -x /isaac-sim/python ]; then CUDA_PY=/isaac-sim/python; fi; '
        'if [ -z "$CUDA_PY" ] && command -v apt-get >/dev/null 2>&1; then '
        "apt-get update >/tmp/blueprint_vast_cuda_probe_apt_update.log 2>&1 && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y python3 >/tmp/blueprint_vast_cuda_probe_apt_install.log 2>&1; "
        'if command -v python3 >/dev/null 2>&1; then CUDA_PY=$(command -v python3); fi; fi; '
        'if [ -z "$CUDA_PY" ]; then '
        "echo BLUEPRINT_VAST_CUDA_RUNTIME_BLOCKED:python_missing; "
        "else $CUDA_PY - <<'PY'\n"
        "import ctypes\n"
        "import ctypes.util\n"
        "import glob\n"
        "names = [ctypes.util.find_library('cudart'), 'libcudart.so']\n"
        "names.extend(glob.glob('/usr/local/cuda*/targets/*/lib/libcudart.so*'))\n"
        "names.extend(glob.glob('/usr/local/lib/python*/dist-packages/nvidia/cuda_runtime/lib/libcudart.so*'))\n"
        "runtime = None\n"
        "for name in dict.fromkeys(item for item in names if item):\n"
        "    try:\n"
        "        runtime = ctypes.CDLL(name)\n"
        "        break\n"
        "    except OSError:\n"
        "        continue\n"
        "if runtime is None:\n"
        "    print('BLUEPRINT_VAST_CUDA_RUNTIME_BLOCKED:cudart_missing', flush=True)\n"
        "    raise SystemExit(2)\n"
        "count = ctypes.c_int()\n"
        "code = int(runtime.cudaGetDeviceCount(ctypes.byref(count)))\n"
        "if code != 0:\n"
        "    print(f'BLUEPRINT_VAST_CUDA_RUNTIME_BLOCKED:cudaGetDeviceCount:{code}', flush=True)\n"
        "    raise SystemExit(3)\n"
        "if count.value < 1:\n"
        "    print('BLUEPRINT_VAST_CUDA_RUNTIME_BLOCKED:no_devices', flush=True)\n"
        "    raise SystemExit(4)\n"
        "print(f'BLUEPRINT_VAST_CUDA_RUNTIME_API_OK:devices={count.value}', flush=True)\n"
        "PY\n"
        "cuda_runtime_rc=$?; fi; "
        "if [ $cuda_runtime_rc -eq 0 ]; then echo BLUEPRINT_VAST_CUDA_RUNTIME_OK; "
        "else echo BLUEPRINT_VAST_CUDA_RUNTIME_EXIT_CODE:$cuda_runtime_rc; fi; "
    )


def gpu_sanity_from_log(text: str, *, require_cuda_runtime: bool) -> dict[str, Any]:
    nvidia_visible = "BLUEPRINT_VAST_GPU_SANITY_OK" in text and "NVIDIA-SMI" not in text
    nvidia_smi_ok = "BLUEPRINT_VAST_GPU_SANITY_OK" in text and not re.search(
        r"nvidia-smi: command not found|failed because it couldn't communicate",
        text,
        flags=re.IGNORECASE,
    )
    cuda_runtime_ok = (
        "BLUEPRINT_VAST_CUDA_RUNTIME_OK" in text
        and "BLUEPRINT_VAST_CUDA_RUNTIME_BLOCKED" not in text
    )
    blockers: list[str] = []
    if not nvidia_smi_ok:
        blockers.append("vast_gpu_sanity_output_missing_or_nvidia_smi_failed")
    if require_cuda_runtime and not cuda_runtime_ok:
        blockers.append("vast_cuda_runtime_host_image_incompatible")
    return {
        "nvidia_visible": nvidia_visible,
        "nvidia_smi_ok": bool(nvidia_smi_ok),
        "cuda_runtime_ok": bool(cuda_runtime_ok),
        "gpu_ok": bool(nvidia_smi_ok and (not require_cuda_runtime or cuda_runtime_ok)),
        "blockers": blockers,
    }


def build_gpu_sanity_report(
    *,
    schema_version: str,
    generated_at: str,
    instance_id: int,
    selected_offer: Mapping[str, Any],
    log_text: str,
    require_cuda_runtime: bool,
    launch_mode: str,
    disk_gb: int,
    container_log_result: Mapping[str, Any],
    truth_boundaries: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    outcome = gpu_sanity_from_log(log_text, require_cuda_runtime=require_cuda_runtime)
    gpu_ok = bool(outcome["gpu_ok"])
    report = {
        "schema_version": schema_version,
        "generated_at": generated_at,
        "status": "completed" if gpu_ok else "blocked",
        "instance_id": instance_id,
        "selected_offer": dict(selected_offer),
        "nvidia_smi_visible": bool(outcome["nvidia_smi_ok"]),
        "gpu_sanity_proven": gpu_ok,
        "driver_cuda_visibility_checked": True,
        "container_cuda_runtime_required": require_cuda_runtime,
        "container_cuda_runtime_compatible": bool(outcome["cuda_runtime_ok"])
        if require_cuda_runtime
        else None,
        "disk_space_checked": True,
        "network_egress_checked": True,
        "bundle_download_ability_checked": False,
        "launch_mode_used": launch_mode,
        "disk_gb": disk_gb,
        "container_log_result": dict(container_log_result),
        "blockers": list(outcome["blockers"]),
        "proof_boundary": (
            "GPU sanity proves provider GPU visibility and, when required, that the "
            "container CUDA runtime can enumerate a device. It does not prove model "
            "load, simulator execution, or scientific validity."
        ),
        **truth_boundaries,
    }
    report["nvidia_smi_marker_absent_from_error"] = bool(outcome["nvidia_visible"])
    return report, outcome
