"""Production allocator that carries one admitted phone capture through Postshot."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .aws_independent_watchdog_control import arm_aws_watchdog, close_aws_watchdog
from .canonical_3dgs_pipeline import build_canonical_3dgs_execution_plan
from .canonical_3dgs_transport import compile_canonical_3dgs_transport_bundle
from .canonical_3dgs_vast_output import validate_canonical_3dgs_vast_output_bundle
from .capture_reconstruction_publication import publish_postshot_output_bundle
from .capture_reconstruction_downstream import dispatch_postshot_to_evidence_spine
from .common import utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest, canonical_json
from .gpu_render_providers import get_render_provider
from .gpu_render_providers import RenderLaunchSpec
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from .postshot_license_transport import close_postshot_license, stage_postshot_license
from .reconstruction_aws_windows_operation import run_reconstruction_aws_windows_operation
from .reconstruction_gpu_admission import (
    CANONICAL_3DGS_RUNTIME_RESULT_SCHEMA,
    CANONICAL_POSTSHOT_AWS_WINDOWS_ADAPTER_ID,
    REQUEST_SCHEMA_VERSION,
    build_reconstruction_gpu_canary_request,
    collect_reconstruction_vast_preflight,
    prepare_reconstruction_gpu_canary,
)
from .reconstruction_vast_operation import _default_output_fetcher
from .wam_provider_object_store import (
    RUNTIME_DEPENDENCY_URL_FILENAME,
    cleanup_staged_wam_provider_objects,
    close_cached_runtime_dependency_staging,
    stage_cached_runtime_dependency_object_store,
    stage_wam_provider_bundle_object_store,
)


class CapturePostshotAllocatorError(RuntimeError):
    pass


_WINDOWS_RUNTIME_DEPENDENCIES = (
    (
        "nvidia-driver",
        "BLUEPRINT_WINDOWS_NVIDIA_DRIVER_FILE",
        "BLUEPRINT_WINDOWS_NVIDIA_DRIVER_SHA256",
        "BLUEPRINT_WINDOWS_NVIDIA_DRIVER_GET_URL",
    ),
    (
        "postshot-installer",
        "BLUEPRINT_WINDOWS_POSTSHOT_INSTALLER_FILE",
        "BLUEPRINT_WINDOWS_POSTSHOT_INSTALLER_SHA256",
        "BLUEPRINT_WINDOWS_POSTSHOT_INSTALLER_GET_URL",
    ),
    (
        "python-embed",
        "BLUEPRINT_WINDOWS_PYTHON_EMBED_FILE",
        "BLUEPRINT_WINDOWS_PYTHON_EMBED_SHA256",
        "BLUEPRINT_WINDOWS_PYTHON_EMBED_GET_URL",
    ),
    (
        "numpy-wheel",
        "BLUEPRINT_WINDOWS_NUMPY_WHEEL_FILE",
        "BLUEPRINT_WINDOWS_NUMPY_WHEEL_SHA256",
        "BLUEPRINT_WINDOWS_NUMPY_WHEEL_GET_URL",
    ),
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise CapturePostshotAllocatorError(f"expected_json_object:{path.name}")
    return dict(value)


def _sha(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return "sha256:" + digest


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise CapturePostshotAllocatorError(
                f"capture_postshot_immutable_output_conflict:{path.name}"
            )


def _required_env(name: str) -> str:
    value = str(os.environ.get(name) or "").strip()
    if not value:
        raise CapturePostshotAllocatorError(f"capture_postshot_required_env_missing:{name}")
    return value


def _require_exact_clean_checkout(expected_commit: str) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CapturePostshotAllocatorError(
            "capture_postshot_checkout_identity_unavailable"
        ) from exc
    if head != expected_commit:
        raise CapturePostshotAllocatorError("capture_postshot_checkout_commit_mismatch")
    if dirty:
        raise CapturePostshotAllocatorError("capture_postshot_checkout_not_clean")
    return {"source_commit_sha": head, "checkout_clean": True}


def _provider_zero_receipt() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for name in ("runpod", "vast", "digitalocean"):
        try:
            inventory = get_render_provider(name).billable_inventory(name_prefix="")
        except Exception as exc:  # noqa: BLE001 - unknown is not zero
            inventory = {"api_confirmed": False, "live_resource_count": None, "error_type": type(exc).__name__}
        passed = inventory.get("api_confirmed") is True and inventory.get("live_resource_count") == 0
        rows.append({
            "provider": name,
            "zero": passed,
            "live_resource_count": inventory.get("live_resource_count"),
            "api_confirmed": inventory.get("api_confirmed") is True,
        })
        if not passed:
            blockers.append(f"capture_postshot_{name}_provider_zero_unproven")
    return {"status": "passed" if not blockers else "blocked", "providers": rows, "blockers": blockers}


def _staging_kwargs() -> dict[str, Any]:
    return {
        "access_key_id_file": _required_env("BLUEPRINT_WAM_OBJECT_STORE_ACCESS_KEY_ID_FILE"),
        "secret_access_key_file": _required_env(
            "BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY_FILE"
        ),
        "endpoint_url_file": os.environ.get("BLUEPRINT_WAM_OBJECT_STORE_ENDPOINT_URL_FILE"),
        "bucket_file": _required_env("BLUEPRINT_WAM_OBJECT_STORE_BUCKET_FILE"),
        "region_file": os.environ.get("BLUEPRINT_WAM_OBJECT_STORE_REGION_FILE"),
    }


def _stage_windows_runtime_dependencies(
    *, root: Path, expiration_seconds: int
) -> tuple[dict[str, str], list[Path]]:
    """Issue fresh run-local URLs for exact reusable Windows runtime bytes."""

    environment: dict[str, str] = {}
    staged_roots: list[Path] = []
    try:
        for label, file_env, digest_env, url_env in _WINDOWS_RUNTIME_DEPENDENCIES:
            staged_root = root / label
            staged = stage_cached_runtime_dependency_object_store(
                job_dir=staged_root,
                dependency_path=_required_env(file_env),
                expected_sha256=_required_env(digest_env),
                key_prefix=f"blueprint/capture-postshot-runtime/{label}",
                expiration_seconds=expiration_seconds,
            )
            staged_roots.append(staged_root)
            if staged.get("status") != "completed":
                raise CapturePostshotAllocatorError(
                    "capture_postshot_runtime_dependency_staging_blocked:"
                    + label
                    + ":"
                    + ",".join(str(item) for item in staged.get("blockers") or [])
                )
            environment[url_env] = (
                (staged_root / RUNTIME_DEPENDENCY_URL_FILENAME).read_text(encoding="utf-8").strip()
            )
            environment[digest_env] = _required_env(digest_env)
        return environment, staged_roots
    except Exception:
        for staged_root in staged_roots:
            close_cached_runtime_dependency_staging(staged_root)
        raise


def _campaign_from_publication(
    *, publication: Mapping[str, Any], plan: Mapping[str, Any], destination: Path
) -> Path:
    campaign: dict[str, Any] = {
        "schema_version": "canonical_3dgs_campaign_result.v1",
        "status": "candidate_artifacts_published",
        "source_capture_digest": plan["source_capture_digest"],
        "canonical_3dgs_source_admission_digest": plan["canonical_3dgs_source_admission_digest"],
        "primary_arm_id": "postshot-primary",
        "world_frame": plan["world_frame"],
        "metric_scale_status": plan["metric_scale_status"],
        "arms": [{
            "arm_id": "postshot-primary",
            "status": "candidate_artifacts_published",
            "artifacts": list(publication["artifacts"]),
        }],
        "appearance_fidelity_qualified": False,
        "metric_accuracy_qualified": False,
        "collision_suitability_qualified": False,
        "physical_task_success_proven": False,
        "completed_at": utc_now_iso(),
    }
    campaign["campaign_digest"] = canonical_digest(campaign, digest_field="campaign_digest")
    write_json(destination, campaign)
    return destination


def _downstream_dispatcher(
    *,
    root: Path,
    raw_root: Path,
    request: Mapping[str, Any],
    publication: Mapping[str, Any],
):
    def dispatch(*, status: Mapping[str, Any]) -> dict[str, Any]:
        payload = dispatch_postshot_to_evidence_spine(
            capture_id=str(request["capture_id"]),
            capture_digest=str(request["capture_digest"]),
            raw_root=raw_root,
            derived_root=root,
            publication=publication,
        )
        payload["terminal_status_digest"] = status["status_digest"]
        payload["dispatch_digest"] = canonical_digest(payload, digest_field="dispatch_digest")
        _write_immutable_json(root / "downstream_analysis_dispatch.json", payload)
        return payload

    return dispatch


def load_postshot_downstream_dispatch(
    request_path: str | Path,
):
    """Rebuild the downstream callback after a dispatcher process restart."""

    path = Path(request_path).expanduser().resolve(strict=True)
    payload = _read(path)
    digest = payload.get("downstream_request_digest")
    if digest != canonical_digest(payload, digest_field="downstream_request_digest"):
        raise CapturePostshotAllocatorError(
            "capture_postshot_downstream_request_digest_invalid"
        )
    publication = payload.get("publication")
    if not isinstance(publication, Mapping):
        raise CapturePostshotAllocatorError(
            "capture_postshot_downstream_publication_invalid"
        )
    return _downstream_dispatcher(
        root=Path(str(payload["derived_root"])),
        raw_root=Path(str(payload["raw_root"])),
        request={
            "capture_id": payload["capture_id"],
            "capture_digest": payload["capture_digest"],
        },
        publication=publication,
    )


def execute_postshot_capture(
    *,
    request: Mapping[str, Any],
    derived_root: str | Path,
    capture_store_root: str | Path,
    max_spend_usd: float,
    hard_ttl_seconds: int,
    authority_id: str,
    retry_cap: int,
) -> dict[str, Any]:
    """Execute one retry-0 capture and return a publication campaign."""

    if retry_cap != 0:
        raise CapturePostshotAllocatorError("capture_postshot_retry_cap_must_be_zero")
    checkout = _require_exact_clean_checkout(str(request["source_commit_sha"]))
    root = Path(derived_root).expanduser().resolve()
    raw_root = Path(capture_store_root).expanduser().resolve(strict=True)
    preparation = _read(root / "canonical_v32_3dgs_preparation.json")
    dataset = _read(root / "colmap_training_dataset_export_result.json")
    dataset_root = root / "trainer_input" / str(dataset["relative_path"])
    plan = build_canonical_3dgs_execution_plan(
        preparation=preparation,
        dataset=dataset,
        dataset_root=dataset_root,
        source_commit_sha=str(request["source_commit_sha"]),
    )
    write_json(root / "canonical_3dgs_execution_plan.json", plan)

    transport_root = root / "postshot-transport"
    transport_root.mkdir(parents=True, exist_ok=True)
    bundle = transport_root / "canonical_3dgs_transport.zip"
    receipt_path = transport_root / "canonical_3dgs_transport_receipt.json"
    receipt = compile_canonical_3dgs_transport_bundle(
        plan=plan,
        dataset_root=dataset_root,
        bundle_path=bundle,
        receipt_path=receipt_path,
        worker_wheel_path=_required_env("BLUEPRINT_CANONICAL_3DGS_WORKER_WHEEL"),
    )

    staged_bundle_root = root / "object-store" / "input"
    staged_receipt_root = root / "object-store" / "receipt"
    staging_kwargs = _staging_kwargs()

    calibration = next(
        (row["digest"] for row in dataset["output_artifacts"] if row["relative_path"].endswith("sparse/0/cameras.txt")),
        None,
    )
    runtime_digest = _required_env("BLUEPRINT_POSTSHOT_RUNTIME_DIGEST")
    runtime_version = _required_env("BLUEPRINT_POSTSHOT_RUNTIME_VERSION")
    worker_image = _required_env("BLUEPRINT_WINDOWS_WORKER_IMAGE_DIGEST")
    gpu_request = build_reconstruction_gpu_canary_request({
        "schema_version": REQUEST_SCHEMA_VERSION,
        "operation": "trainer_canary",
        "capture_profile": "iphone_arkit_lidar",
        "source_commit_sha": request["source_commit_sha"],
        "worker_image_digest": worker_image,
        "worker_stack_manifest_digest": canonical_digest({"runtime_digest": runtime_digest, "runtime_version": runtime_version, "worker_wheel_digest": receipt["worker_wheel_digest"]}),
        "deterministic_configuration_digest": plan["canonical_3dgs_execution_plan_digest"],
        "operation_request_digest": plan["canonical_3dgs_execution_plan_digest"],
        "operation_input_bundle_digest": receipt["transport_bundle_digest"],
        "source_capture_digest": request["capture_digest"],
        "reconstruction_dataset_digest": dataset["colmap_training_dataset_digest"],
        "frozen_split_digest": dataset["frozen_split_digest"],
        "calibration_digest": calibration,
        "expected_runtime_result_schema": CANONICAL_3DGS_RUNTIME_RESULT_SCHEMA,
        "candidate_may_read_hidden_heldout": False,
        "trainer_may_grade_heldout": False,
        "max_spend_usd": max_spend_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "retry_cap": 0,
        "authority_id": authority_id,
        "requested_execution_adapter_id": CANONICAL_POSTSHOT_AWS_WINDOWS_ADAPTER_ID,
        "proof_effect": "none",
    })
    request_path = root / "reconstruction_gpu_canary_request.json"
    write_json(request_path, gpu_request)

    provider_zero = _provider_zero_receipt()
    write_json(root / "non_aws_provider_zero.json", provider_zero)
    if provider_zero["status"] != "passed":
        raise CapturePostshotAllocatorError("capture_postshot_conflicting_provider_resources")

    name_prefix = f"blueprint-postshot-{str(request['capture_id'])[:64]}"
    watchdog_handoff, watchdog_process = arm_aws_watchdog(
        job_dir=root / "watchdog", name_prefix=name_prefix, hard_ttl_seconds=hard_ttl_seconds
    )
    provider = get_render_provider("aws")
    max_hourly = float(_required_env("BLUEPRINT_AWS_MAX_HOURLY_RATE_USD"))
    container_bytes = int(_required_env("BLUEPRINT_RECONSTRUCTION_CONTAINER_DISK_BYTES"))
    probe_request = provider.build_request(
        RenderLaunchSpec(
            name=name_prefix,
            image=worker_image,
            env={"BLUEPRINT_WORKER_HARD_TTL_SECONDS": str(hard_ttl_seconds)},
            bootstrap_argv=[],
            entrypoint=[],
            container_disk_gb=max(100, container_bytes // 1024**3),
            volume_gb=0,
            max_hourly_rate_usd=max_hourly,
            # AWS reports the A10G's marketed 24 GB as 22,888 MiB. Keep the
            # admission above 22 GiB while avoiding a false rejection of the
            # explicitly configured compatible g5.xlarge worker.
            min_gpu_ram_mb=22_000,
            requires_rtx=False,
        ),
        root / "aws_preflight",
    )
    preflight = collect_reconstruction_vast_preflight(
        name_prefix=name_prefix,
        container_disk_bytes=container_bytes,
        watchdog=watchdog_handoff,
        conflicting_owner_present=False,
        capacity_probe=lambda _request: provider.capacity_preflight(probe_request),
        inventory_probe=lambda prefix: provider.billable_inventory(name_prefix=prefix),
        max_hourly_rate_usd=max_hourly,
        provider_name="aws",
    )
    preflight_path = root / "reconstruction_gpu_preflight.json"
    write_json(preflight_path, preflight)
    admission_path = root / "reconstruction_gpu_admission.json"
    bound_path = root / "reconstruction_gpu_bound_request.json"
    adapter_path = root / "reconstruction_gpu_adapter_result.json"
    admission = prepare_reconstruction_gpu_canary(
        request_path=request_path,
        preflight_path=preflight_path,
        admission_out=admission_path,
        bound_request_out=bound_path,
        adapter_output=adapter_path,
        provider="aws",
        expected_source_commit=str(request["source_commit_sha"]),
        checkout_source_commit=str(checkout["source_commit_sha"]),
        checkout_clean=bool(checkout["checkout_clean"]),
        max_spend_usd=max_spend_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        retry_cap=0,
        authority_id=authority_id,
        execute=True,
        execution_adapter_id=CANONICAL_POSTSHOT_AWS_WINDOWS_ADAPTER_ID,
    )

    license_stage: dict[str, Any] | None = None
    runtime_environment: dict[str, str] = {}
    runtime_staged_roots: list[Path] = []
    staged_object_roots: list[Path] = []
    operation: dict[str, Any] = {}
    watchdog_close: dict[str, Any] = {}
    cleanup: list[dict[str, Any]] = []
    try:
        lane = build_paid_lane_admission(
            resource_class="gpu_render", blockers=list(admission.get("blockers") or [])
        )
        grant = require_paid_resource_admission(
            lane,
            resource_class="gpu_render",
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        )
        staged_object_roots.append(staged_bundle_root)
        staged_bundle = stage_wam_provider_bundle_object_store(
            job_dir=staged_bundle_root,
            bundle_path=bundle,
            key_prefix="blueprint/capture-postshot",
            expiration_seconds=hard_ttl_seconds,
            **staging_kwargs,
        )
        staged_object_roots.append(staged_receipt_root)
        staged_receipt = stage_wam_provider_bundle_object_store(
            job_dir=staged_receipt_root,
            bundle_path=receipt_path,
            key_prefix="blueprint/capture-postshot-receipts",
            output_content_type="application/json",
            expiration_seconds=hard_ttl_seconds,
            **staging_kwargs,
        )
        staging_blockers = [
            *staged_bundle.get("blockers", []),
            *staged_receipt.get("blockers", []),
        ]
        if staging_blockers:
            raise CapturePostshotAllocatorError(
                "capture_postshot_staging_blocked:" + ",".join(staging_blockers)
            )
        runtime_environment, runtime_staged_roots = _stage_windows_runtime_dependencies(
            root=root / "runtime-dependencies",
            expiration_seconds=hard_ttl_seconds,
        )
        license_stage = stage_postshot_license(
            job_dir=root / "license",
            license_file=_required_env("BLUEPRINT_POSTSHOT_LICENSE_FILE"),
            expiration_seconds=hard_ttl_seconds,
        )
        transient_environment = {
            **runtime_environment,
            "BLUEPRINT_POSTSHOT_LICENCE_GET_URL": str(license_stage["get_url"]),
            "BLUEPRINT_POSTSHOT_LICENCE_DELETE_URL": str(license_stage["delete_url"]),
        }
        previous = {key: os.environ.get(key) for key in transient_environment}
        os.environ.update(transient_environment)
        try:
            operation = run_reconstruction_aws_windows_operation(
                bound_request=_read(bound_path),
                preflight=preflight,
                job_dir=root / "reconstruction_aws_windows_operation",
                output_bundle_get_url=(staged_bundle_root / "provider_output_get_url.txt")
                .read_text()
                .strip(),
                input_bundle_get_url=(staged_bundle_root / "provider_bundle_url.txt")
                .read_text()
                .strip(),
                input_receipt_get_url=(staged_receipt_root / "provider_bundle_url.txt")
                .read_text()
                .strip(),
                input_receipt_file_digest=_sha(receipt_path),
                output_bundle_put_url=(staged_bundle_root / "provider_output_put_url.txt")
                .read_text()
                .strip(),
                progress_put_url=(staged_receipt_root / "provider_output_put_url.txt")
                .read_text()
                .strip(),
                progress_get_url=(staged_receipt_root / "provider_output_get_url.txt")
                .read_text()
                .strip(),
                progress_observer=lambda value: write_json(
                    root / "postshot_live_progress.json", dict(value)
                ),
                provider=provider,
                allocator_admission=admission,
                paid_resource_admission_grant=grant,
                name_prefix=name_prefix,
                hard_ttl_seconds=hard_ttl_seconds,
                output_fetcher=_default_output_fetcher,
                output_validator=validate_canonical_3dgs_vast_output_bundle,
            )
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    finally:
        if license_stage is not None:
            cleanup.append(close_postshot_license(staged=license_stage, job_dir=root / "license"))
        cleanup.extend(
            close_cached_runtime_dependency_staging(staged_root)
            for staged_root in runtime_staged_roots
        )
        cleanup.extend(
            cleanup_staged_wam_provider_objects(staged_root, **staging_kwargs)
            for staged_root in staged_object_roots
        )
        watchdog_close = close_aws_watchdog(job_dir=root / "watchdog", process=watchdog_process)

    if operation.get("status") != "completed":
        return {"status": "blocked", "blockers": operation.get("blockers") or ["capture_postshot_worker_failed"], "provider_mutations_performed": operation.get("provider_mutations_performed", 0)}
    if watchdog_close.get("status") != "provider_terminal" or any(row.get("status") not in {"completed", "closed"} for row in cleanup):
        return {"status": "blocked", "blockers": ["capture_postshot_cleanup_or_provider_zero_unproven"], "provider_mutations_performed": operation.get("provider_mutations_performed", 0)}

    publication = publish_postshot_output_bundle(
        output_bundle=root / "reconstruction_aws_windows_operation" / "output_bundle.zip",
        capture_id=str(request["capture_id"]),
        capture_digest=str(request["capture_digest"]),
        bucket_name=_required_env("BLUEPRINT_CAPTURE_RECONSTRUCTION_PUBLICATION_BUCKET"),
        publication_root=root / "publication",
    )
    campaign_path = _campaign_from_publication(
        publication=publication, plan=plan, destination=root / "canonical_3dgs_campaign_result.json"
    )
    downstream_request: dict[str, Any] = {
        "schema_version": "capture_reconstruction_downstream_request.v1",
        "capture_id": request["capture_id"],
        "capture_digest": request["capture_digest"],
        "raw_root": str(raw_root),
        "derived_root": str(root),
        "publication": publication,
    }
    downstream_request["downstream_request_digest"] = canonical_digest(
        downstream_request, digest_field="downstream_request_digest"
    )
    downstream_request_path = root / "downstream_analysis_request.json"
    _write_immutable_json(downstream_request_path, downstream_request)
    return {
        "status": "completed",
        "admission_digest": admission["admission_digest"],
        "operation_receipt_digest": canonical_digest(operation),
        "campaign_path": str(campaign_path),
        "completed_at": utc_now_iso(),
        "provider_mutations_performed": operation.get("provider_mutations_performed", 0),
        "provider_zero": watchdog_close,
        "downstream_dispatch_request_path": str(downstream_request_path),
        "downstream_dispatch": load_postshot_downstream_dispatch(
            downstream_request_path
        ),
    }


def production_allocator(**kwargs: Any) -> dict[str, Any]:
    """Dispatcher-shaped entry point; rejects unsupported comparison spend."""

    arms = list(kwargs.pop("arms", []))
    if arms != ["postshot-primary"]:
        return {"status": "blocked", "blockers": ["capture_postshot_only_primary_arm_supported"], "provider_mutations_performed": 0}
    kwargs.pop("capture_digest", None)
    derived_root = kwargs.pop("dataset_root")
    return execute_postshot_capture(derived_root=derived_root, **kwargs)


__all__ = [
    "CapturePostshotAllocatorError",
    "execute_postshot_capture",
    "load_postshot_downstream_dispatch",
    "production_allocator",
]
