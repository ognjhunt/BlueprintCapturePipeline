"""Fail-closed GR00T + OSCAR RunPod GPU canary launcher.

The provider adapter is unreachable until the exact release, model cache,
existing network volume, GPU/CUDA constraints, budget, and already-armed
watchdog produce an admitted record.  This launcher is intentionally scoped to
the startup canary; it is not an image builder or a customer cold-start path.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import write_json
from .groot_oscar_infrastructure_admission import (
    SERVE_SCHEMA_VERSION,
    build_runpod_serve_plane_admission,
)
from .paid_resource_admission import require_paid_resource_admission
from .runpod_provider_adapter import (
    RUNPOD_IMAGE_STARTUP_CANARY_MODE,
    run_runpod_provider_adapter,
)


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected_json_object:{path}")
    return dict(value)


def bind_canary_request(
    *, request: Mapping[str, Any], admission: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the adapter request to the already-admitted immutable tuple."""

    result = deepcopy(dict(request))
    shape = result.get("provider_request_shape")
    shape = shape if isinstance(shape, dict) else {}
    image = shape.get("image")
    image = image if isinstance(image, dict) else {}
    configured = str(image.get("configured_image_ref") or "").strip()
    admitted_image = str(admission.get("release_image_ref") or "").strip()
    blockers: list[str] = []
    if configured != admitted_image:
        blockers.append("runpod_request_release_image_differs_from_admission")
    gpu = shape.get("gpu")
    gpu = gpu if isinstance(gpu, dict) else {}
    admitted_gpu = str(admission.get("gpu_type_id") or "").strip()
    configured_gpu = str(
        gpu.get("preferred_gpu_type_id") or gpu.get("preferred_gpu_class") or ""
    ).strip()
    if configured_gpu and configured_gpu != admitted_gpu:
        blockers.append("runpod_request_gpu_differs_from_admission")
    shape["network_volume_id"] = admission.get("network_volume_id")
    shape["data_center_id"] = admission.get("data_center_id")
    shape["allowed_cuda_versions"] = [admission.get("required_cuda_version")]
    gpu["preferred_gpu_type_id"] = admitted_gpu
    gpu["provider_gpu_priority"] = [admitted_gpu]
    shape["gpu"] = gpu
    shape["image"] = image
    result["provider_request_shape"] = shape
    return {
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "request": result,
    }


def prepare_canary_launch(
    *,
    request: Mapping[str, Any],
    release: Mapping[str, Any],
    model_cache: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    volume = preflight.get("volume")
    runtime = preflight.get("runtime")
    spend = preflight.get("spend")
    admission = build_runpod_serve_plane_admission(
        release=release,
        model_cache=model_cache,
        volume=volume if isinstance(volume, Mapping) else {},
        runtime=runtime if isinstance(runtime, Mapping) else {},
        spend=spend if isinstance(spend, Mapping) else {},
    )
    if preflight.get("status") != "verified":
        admission = {
            **admission,
            "status": "blocked",
            "blockers": sorted(
                set([*admission.get("blockers", []), "runpod_preflight_bundle_not_verified"])
            ),
        }
    bound = bind_canary_request(request=request, admission=admission)
    blockers = list(admission.get("blockers") or []) + list(bound["blockers"])
    return {
        "status": "admitted" if not blockers and admission["status"] == "admitted" else "blocked",
        "blockers": sorted(set(blockers)),
        "admission": admission,
        "bound_request": bound["request"],
        "provider_mutations_performed": 0,
    }


def run_canary(
    *,
    provider_launch_request: str | Path,
    release_evidence: str | Path,
    model_cache_evidence: str | Path,
    preflight_bundle: str | Path,
    admission_out: str | Path,
    bound_request_out: str | Path,
    adapter_output: str | Path,
    pod_name: str,
    execute: bool,
) -> dict[str, Any]:
    """Run the adapter only through the canonical GPU-canary allocator."""

    preflight = _read(preflight_bundle)
    prepared = prepare_canary_launch(
        request=_read(provider_launch_request),
        release=_read(release_evidence),
        model_cache=_read(model_cache_evidence),
        preflight=preflight,
    )
    spend = preflight.get("spend")
    spend = spend if isinstance(spend, Mapping) else {}
    watchdog_prefix = str(spend.get("watchdog_pod_name_prefix") or "").strip()
    if watchdog_prefix and not pod_name.startswith(watchdog_prefix):
        prepared = {
            **prepared,
            "status": "blocked",
            "blockers": sorted(
                set(
                    [
                        *prepared.get("blockers", []),
                        "runpod_canary_pod_name_outside_watchdog_scope",
                    ]
                )
            ),
        }
    write_json(Path(admission_out), prepared)
    if prepared["status"] != "admitted":
        return prepared
    require_paid_resource_admission(
        prepared["admission"],
        resource_class="gpu_canary",
        expected_schema_version=SERVE_SCHEMA_VERSION,
    )
    write_json(Path(bound_request_out), prepared["bound_request"])
    adapter = run_runpod_provider_adapter(
        provider_launch_request_path=bound_request_out,
        output_path=adapter_output,
        mode=RUNPOD_IMAGE_STARTUP_CANARY_MODE if execute else "dry-run",
        allow_runpod_api_call=execute,
        pod_name=pod_name,
        gpu_type_id=prepared["admission"]["gpu_type_id"],
    )
    if execute and adapter.get("status") == "submitted":
        response = adapter.get("runpod_response")
        response = response if isinstance(response, Mapping) else {}
        pod_id = str(response.get("id") or "").strip()
        write_json(
            Path(adapter_output).resolve().parent / "warm_serve_pod.json",
            {
                "schema_version": "groot_oscar_runpod_canary_allocation.v1",
                "status": "allocated" if pod_id else "allocation_ambiguous",
                "pod_id": pod_id or None,
                "pod_name": pod_name,
                "release_image_ref": prepared["admission"]["release_image_ref"],
            },
        )
    return dict(adapter)


def main(argv: Sequence[str] | None = None) -> int:
    """Hard-disable the legacy mutation entrypoint.

    Imports remain for compatibility and tests, but allocation is only exposed
    by ``blueprint-allocate-gpu-canary``.
    """

    del argv
    print("legacy_gpu_canary_launcher_disabled:use_blueprint-allocate-gpu-canary")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
