"""ADP-009B/day14: controller dispatch of estimated geometry through the allocator.

One deployment profile supplies pinned runtime files, not per-capture shell
commands. The existing allocator owns the GPU, watchdog, retrieval and teardown.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from .common import sha256_file, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .gpu_render_providers import get_render_provider
from .reconstruction_gpu_admission import collect_reconstruction_vast_preflight
from .reconstruction_gpu_operation_bundle import build_canary_request_from_operation_bundle
from .reconstruction_vast_operation import NAME_PREFIX, _canonical_receipt_file, replay_reconstruction_vast_operation, reconstruction_resource_name
from .vast_independent_watchdog_control import arm_independent_vast_watchdog, close_independent_vast_watchdog
from .wam_provider_object_store import stage_wam_provider_bundle_object_store, cleanup_staged_wam_provider_objects
from .website_mapanything_operation import compile_input_bundle
from .website_scene_geometry import load_website_geometry_result
from .website_task_context import reserve_website_preparation_spend, load_website_scene_sponsorship
from .website_vast_billing import settle_prior_website_vast_attempts


def load_profile(*, source_commit: str, profile_path: str | Path | None = None) -> dict[str, Any]:
    path = profile_path or os.getenv("BLUEPRINT_WEBSITE_MAPANYTHING_PROFILE", "")
    if not path:
        raise ValueError("website_mapanything_runtime_profile_missing")
    value = json.loads(Path(path).read_text())
    if value.get("schema_version") != "website_mapanything_runtime.v1" or value.get("source_commit") != source_commit:
        raise ValueError("website_mapanything_runtime_release_mismatch")
    for key in ("maximum_cost_usd", "max_hourly_rate_usd"):
        number = value.get(key)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
            raise ValueError("website_mapanything_runtime_budget_invalid")
    ttl = value.get("hard_ttl_seconds")
    if type(ttl) is not int or not 120 <= ttl <= 3600:
        raise ValueError("website_mapanything_runtime_ttl_invalid")
    if value["max_hourly_rate_usd"] * ttl / 3600 > value["maximum_cost_usd"]:
        raise ValueError("website_mapanything_runtime_budget_below_ttl")
    if type(value.get("minimum_gpu_ram_mb")) is not int or value["minimum_gpu_ram_mb"] < 24000:
        raise ValueError("website_mapanything_runtime_memory_invalid")
    rows = value.get("runtime_files")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("website_mapanything_runtime_files_missing")
    paths = []
    for row in rows:
        file = Path(row["path"])
        if not file.is_absolute() or file.is_symlink() or not file.is_file() or "sha256:" + sha256_file(file) != row["digest"]:
            raise ValueError("website_mapanything_runtime_file_changed")
        paths.append(file)
    if len({p.name for p in paths}) != 4 or sum(p.suffix == ".whl" for p in paths) != 3:
        raise ValueError("website_mapanything_runtime_files_invalid")
    return value


def _reuse(root: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    request = json.loads((root / "bound-request.json").read_text())
    operation_root = root / "reconstruction_vast_operation"
    replay = replay_reconstruction_vast_operation(job_dir=operation_root, bound_request=request)
    if replay["status"] != "replay_verified":
        raise ValueError("website_mapanything_existing_attempt_requires_reconciliation")
    execution = json.loads((operation_root / "reconstruction_vast_operation_execution.json").read_text())
    receipt = json.loads((operation_root / "validated_output_bundle_receipt.json").read_text())
    bundle = next(path for path in (operation_root / "retrieval_attempts").glob("output_*.zip")
                  if "sha256:" + sha256_file(path) == execution["operation_output_bundle_digest"])
    destination = root / "verified-output"
    destination.mkdir(exist_ok=True)
    # Replay validates the complete portable-path inventory and all member bytes.
    with zipfile.ZipFile(bundle) as archive:
        for row in receipt["artifact_members"]:
            target = destination / row["archive_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("website_mapanything_output_path_invalid")
            if target.exists():
                if "sha256:" + sha256_file(target) != row["digest"]:
                    raise ValueError("website_mapanything_output_changed")
            else:
                with archive.open(row["archive_path"]) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    return load_website_geometry_result(manifest_path=destination / "artifacts/source_geometry.json", inputs=inputs)


def _failed_attempt_safe_to_retry(root: Path, *, source_commit: str) -> bool:
    """Admit one new release only after the old rental has failed and torn down."""
    try:
        request = json.loads((root / "request.json").read_text())
        operation = root / "reconstruction_vast_operation"
        execution = json.loads((operation / "reconstruction_vast_operation_execution.json").read_text())
        provider_zero = json.loads((operation / "provider_zero_verification.json").read_text())
        teardown = json.loads((operation / "teardown_receipt.json").read_text())
    except (OSError, ValueError, KeyError):
        return False
    return (
        request.get("source_commit_sha") != source_commit
        and execution.get("status") == "failed"
        and execution.get("request_digest") == request.get("request_digest")
        and execution.get("provider_zero_verified") is True
        and "reconstruction_vast_operation_output_not_accepted" in execution.get("blockers", [])
        and execution.get("execution_result_digest") == canonical_digest(execution, digest_field="execution_result_digest")
        and execution.get("provider_zero_digest") == provider_zero.get("provider_zero_digest")
        and provider_zero.get("provider_zero_digest") == canonical_digest(provider_zero, digest_field="provider_zero_digest")
        and provider_zero.get("status") == "PASS"
        and execution.get("teardown_receipt_digest") == teardown.get("teardown_receipt_digest")
        and teardown.get("teardown_receipt_digest") == canonical_digest(teardown, digest_field="teardown_receipt_digest")
        and teardown.get("provider_zero_verified") is True
    )


def dispatch_geometry(*, input_manifest: Path, output_root: Path, task_context: Mapping[str, Any],
                      source_commit: str, allocate: Callable[..., Mapping[str, Any]]) -> dict[str, Any]:
    inputs = json.loads(input_manifest.read_text())
    first_root = output_root / "controller_geometry"
    first_root.mkdir(parents=True, exist_ok=True)
    with (first_root / "dispatch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("website_mapanything_controller_already_running") from exc
        for attempt_index in range(3):
            root = first_root if attempt_index == 0 else output_root / f"controller_geometry_retry_{attempt_index}"
            root.mkdir(parents=True, exist_ok=True)
            state_path = root / "dispatch.json"
            if not state_path.is_file():
                break
            state = json.loads(state_path.read_text())
            if state["input_digest"] != inputs["digest"] or state["task_context_digest"] != task_context["context_digest"]:
                raise ValueError("website_mapanything_controller_inputs_changed")
            try:
                return _reuse(root, inputs)
            except ValueError as exc:
                if (str(exc) != "website_mapanything_existing_attempt_requires_reconciliation"
                        or attempt_index == 2
                        or not _failed_attempt_safe_to_retry(root, source_commit=source_commit)):
                    raise
        if attempt_index:
            settle_prior_website_vast_attempts(output_root=output_root,
                attempted_count=attempt_index, task_context=dict(task_context))
        profile = load_profile(source_commit=source_commit)
        binding = canonical_digest({"input_digest": inputs["digest"], "runtime": profile,
                                    "task_context_digest": task_context["context_digest"]})
        authority = load_website_scene_sponsorship(task_context=task_context, now=time.time())
        if authority["expires_at_epoch"] < time.time() + profile["hard_ttl_seconds"] + 120:
            raise ValueError("website_mapanything_sponsorship_expires_before_teardown")
        receipt = compile_input_bundle(input_manifest=input_manifest, output_root=root / "bundle",
            source_commit_sha=source_commit, worker_image_digest=profile["worker_image_digest"],
            remote_processing_authorization_digest=authority["authority_digest"],
            runtime_files=[Path(row["path"]) for row in profile["runtime_files"]])
        request = build_canary_request_from_operation_bundle(operation_bundle=receipt, request_fields={
            "schema_version": "reconstruction_gpu_canary_request.v1", "worker_stack_manifest_digest": canonical_digest(profile["runtime_files"]),
            "deterministic_configuration_digest": binding, "max_spend_usd": profile["maximum_cost_usd"],
            "hard_ttl_seconds": profile["hard_ttl_seconds"], "retry_cap": 0,
            "authority_id": "website-" + binding[7:31], "proof_effect": "none",
            "candidate_may_read_hidden_heldout": False, "trainer_may_grade_heldout": False})
        write_json(root / "request.json", request)
        canonical_receipt, _ = _canonical_receipt_file(root, receipt)
        name = reconstruction_resource_name(request["operation"], request["request_digest"])
        handoff, handle = arm_independent_vast_watchdog(job_dir=root,
            max_live_minutes=math.ceil(profile["hard_ttl_seconds"] / 60) + 2,
            generated_at=utc_now_iso(), pod_name_prefix=NAME_PREFIX, resource_name_exact=name)
        if handle is None:
            raise ValueError("website_mapanything_watchdog_not_armed")
        invoked, result = False, {}
        staged = []
        try:
            provider = get_render_provider("vast")
            preflight = collect_reconstruction_vast_preflight(name_prefix=NAME_PREFIX, container_disk_bytes=100 * 1024**3,
                watchdog={"status": "armed", "independent_process": True, "pid": handoff["watchdog_pid"],
                          "deadline_epoch": handoff["watchdog_deadline_epoch"], "name_prefix": name,
                          "watchdog_out_dir": handoff["watchdog_out_dir"]},
                conflicting_owner_present=False, capacity_probe=provider.capacity_preflight,
                inventory_probe=lambda prefix: provider.billable_inventory(name_prefix=prefix),
                max_hourly_rate_usd=profile["max_hourly_rate_usd"], minimum_gpu_ram_mb=profile["minimum_gpu_ram_mb"])
            write_json(root / "preflight.json", preflight)
            if preflight["status"] != "verified":
                raise ValueError("website_mapanything_preflight_blocked:" + ",".join(preflight["blockers"]))
            admission, _ = reserve_website_preparation_spend(task_context=task_context, binding_digest=binding,
                maximum_cost_usd=profile["maximum_cost_usd"], request_count=1, resource_class="gpu_render", provider="vast")
            write_json(root / "controller_admission.json", admission)
            bundle = root / "bundle" / receipt["bundle_artifact_reference"]
            if (not bundle.is_file() or "sha256:" + sha256_file(bundle) != receipt["operation_input_bundle_digest"]
                    or bundle.stat().st_size != receipt["bundle_bytes"]):
                raise ValueError("website_mapanything_exact_bundle_changed")
            for folder, artifact in (("transport", bundle),
                                     ("receipt-transport", canonical_receipt)):
                path = root / folder
                staged.append(path)
                transfer = stage_wam_provider_bundle_object_store(job_dir=path, bundle_path=artifact,
                    expiration_seconds=profile["hard_ttl_seconds"] + 600)
                if transfer.get("status") != "completed":
                    raise ValueError("website_mapanything_transport_not_ready")
            args = SimpleNamespace(provider="vast", execute=True, expected_source_commit=source_commit,
                provider_launch_request=str(root / "request.json"), preflight_bundle=str(root / "preflight.json"),
                admission_out=str(root / "admission.json"), bound_request_out=str(root / "bound-request.json"),
                adapter_output=str(root / "adapter-result.json"), reconstruction_max_spend_usd=profile["maximum_cost_usd"],
                reconstruction_hard_ttl_seconds=profile["hard_ttl_seconds"], reconstruction_retry_cap=0,
                reconstruction_authority_id=request["authority_id"],
                reconstruction_operation_bundle_receipt=str(canonical_receipt),
                reconstruction_operation_receipt_url_file=str(root / "receipt-transport/provider_bundle_url.txt"),
                provider_bundle_url_file=str(root / "transport/provider_bundle_url.txt"),
                provider_output_put_url_file=str(root / "transport/provider_output_put_url.txt"),
                provider_output_get_url_file=str(root / "transport/provider_output_get_url.txt"))
            # Write before invoking the allocator. A crash cannot become a second rental.
            with state_path.open("x") as stream:
                json.dump({"input_digest": inputs["digest"], "task_context_digest": task_context["context_digest"],
                           "request_digest": request["request_digest"]}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            invoked = True
            result = dict(allocate(args, checkout_commit=source_commit))
            if result.get("status") != "completed":
                raise ValueError("website_mapanything_worker_incomplete:" + ",".join(result.get("blockers") or []))
            return _reuse(root, inputs)
        finally:
            # Preserve the watchdog and transport whenever allocation or teardown is uncertain.
            safe = not invoked or result.get("provider_zero_verified") is True
            close_independent_vast_watchdog(job_dir=root, handle=handle,
                instance_ids=[int(result["instance_id"])] if result.get("instance_id") else [],
                provider_teardown_completed=result.get("provider_zero_verified") is True,
                provider_allocation_impossible=not invoked, wait_seconds=30)
            if safe:
                for path in staged:
                    cleanup_staged_wam_provider_objects(job_dir=path)
