"""Durable explicit batch queue and verified operator-owned robot model store."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .deployment_embodiment_preparation_contract import (
    MODEL_ROOT, PROBE_KIND, RESULT_FILENAME, STATE_ROOT, WORK_ROOT, MAX_OUTPUT_BYTES, read, require, sealed, sha,
    validate_preparation_result, validate_request, write_once,
)
from .deployment_embodiment_preparation_bundle import RECEIPT, build_bundle, load_verified_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]


def source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def enqueue(request: dict, *, root: Path = STATE_ROOT, commit=None) -> dict:
    request = validate_request(request)
    require(request["release_commit"] == (commit or source_commit()), "installed_release_mismatch")
    require(not any(p.is_symlink() for p in (root, *root.parents)), "queue_path_invalid")
    name = request["authorization_id"] + ".json"
    write_once(root / "authorizations" / name, request)
    write_once(root / "pending" / name, {"authorization_id": request["authorization_id"],
                                         "request_digest": request["request_digest"]})
    return {"status": "queued", "authorization_id": request["authorization_id"],
            "request_digest": request["request_digest"], "provider_allocations_performed": 0}


def runtime_source_receipt() -> Path:
    from .team_candidate_quick10_registry import load_registry
    from .native_task_runtime_source_packet import verify_native_task_runtime_source_packet
    registry = load_registry()
    require(isinstance(registry, dict), "operator_registry_missing")
    raw = (registry.get("derived_profiles") or {}).get("packet_inputs", {}).get("runtime_source_receipt_path")
    require(isinstance(raw, str) and Path(raw).is_absolute(), "operator_runtime_source_missing")
    path = Path(raw)
    require(not any(p.is_symlink() for p in (path, *path.parents)), "operator_runtime_source_path_invalid")
    verify_native_task_runtime_source_packet(path)
    return path


def workspace_for(job: Path) -> Path:
    return (WORK_ROOT if job.parent.parent == STATE_ROOT else job.parent.parent / "work") / job.name


def allocator_argv(request: dict, job: Path) -> list[str]:
    return ["gpu-canary", "--provider", "vast", "--probe-kind", PROBE_KIND,
            "--embodiment-preparation-request", str(job.parent.parent / "authorizations" / (request["authorization_id"] + ".json")),
            "--embodiment-preparation-bundle", str(workspace_for(job) / "bundle" / RECEIPT),
            "--embodiment-preparation-job-dir", str(workspace_for(job) / "allocator"),
            "--admission-out", str(job / "paid-admission.json"),
            "--adapter-output", str(job / "allocator_result.json"), "--execute"]


def retain_batch_diagnostic(*, body: Path, row: dict, binding: dict, request: dict, store: Path) -> dict:
    """Keep this batch's verified diagnostic separate from reusable model bytes."""
    from .decision_evidence_contracts import canonical_digest
    from .immutable_directory_publication import publish_staged_immutable_directory

    names = ("native-controller-process.v1.json", "embodiment_native_controller_smoke.v1.json")
    reported = row.get("native_controller_smoke") or {}
    diagnostic = {"status": "failed", "verification_status": "unverified", "process_digest": None,
                  "smoke_digest": None, "reasons": [], "claim_ceiling": "native_controller_diagnostic_only",
                  "candidate_policy_queried": False, "task_executed": False, "physical_success_proven": False}
    try:
        require(isinstance(reported, dict), "diagnostic_summary_invalid")
        process = read(body / names[0])
        require(process.get("schema_version") == "embodiment_native_controller_process.v1"
                and process.get("process_digest") == canonical_digest(process, digest_field="process_digest")
                and process.get("status") in {"passed", "failed"}
                and all(process.get(key) is False for key in
                        ("candidate_policy_queried", "task_executed", "physical_success_proven")), "diagnostic_process_invalid")
        require(all(reported.get(key) == process.get(key) for key in
                    ("status", "process_digest", "smoke_digest", "reasons")), "diagnostic_summary_changed")
        if process.get("smoke_digest") is not None:
            native = read(body / names[1])
            provenance = binding["model_record"]["source_import_provenance"]
            expected = {"embodiment_id": row["embodiment_id"],
                        "embodiment_digest": binding["robot_embodiment"]["embodiment_digest"],
                        "binding_digest": canonical_digest(binding), "model_sha256": binding["model_record"]["sha256"],
                        "source_bundle_digest": provenance["source_bundle_digest"], "import_digest": provenance["import_digest"],
                        "configuration_digest": binding["configuration_digest"]}
            require(native.get("schema_version") == "embodiment_native_controller_smoke.v1"
                    and native.get("smoke_digest") == process["smoke_digest"]
                    == canonical_digest(native, digest_field="smoke_digest")
                    and all(native.get(key) == value for key, value in expected.items())
                    and native.get("claim_ceiling") == "native_controller_diagnostic_only"
                    and all(native.get(key) is False for key in
                            ("candidate_policy_queried", "task_executed", "physical_success_proven"))
                    and native.get("status") == process["status"] and native.get("reasons") == process["reasons"],
                    "diagnostic_native_binding_changed")
        else:
            require(process["status"] == "failed", "diagnostic_success_without_native_receipt")
        diagnostic.update({key: process[key] for key in ("status", "process_digest", "smoke_digest", "reasons")},
                          verification_status="verified")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        diagnostic["reasons"] = ["diagnostic_receipt_unverified:" + str(exc)[:240]]
        if isinstance(reported, dict):
            diagnostic["reported_reasons"] = reported.get("reasons", [])
    destination = store / "batches" / request["authorization_id"] / row["embodiment_id"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not any(p.is_symlink() for p in (destination, *destination.parents)), "diagnostic_store_path_invalid")
    files = [{"path": name, "sha256": sha(body / name), "size_bytes": (body / name).stat().st_size}
             for name in names if (body / name).is_file()]
    retained = sealed({"schema_version": "embodiment_batch_diagnostic.v1", "request_digest": request["request_digest"],
        "binding_digest": canonical_digest(binding), "embodiment_id": row["embodiment_id"],
        "diagnostic": diagnostic, "files": files}, "diagnostic_digest")
    if destination.exists():
        require(read(destination / "diagnostic.json") == retained
                and all(sha(destination / file["path"]) == file["sha256"] for file in files), "retained_diagnostic_changed")
    else:
        with tempfile.TemporaryDirectory(prefix=".diagnostic-", dir=destination.parent) as temporary:
            staged = Path(temporary)
            for file in files:
                shutil.copyfile(body / file["path"], staged / file["path"])
            write_once(staged / "diagnostic.json", retained)
            publish_staged_immutable_directory(staging=staged, destination=destination,
                manifest_name="diagnostic.json", output_exists_code="embodiment_preparation_diagnostic_exists")
    return {**diagnostic, "diagnostic_path": str(destination / "diagnostic.json"),
            "diagnostic_digest": retained["diagnostic_digest"],
            "receipt_paths": {file["path"]: str(destination / file["path"]) for file in files}}


def promote(*, request: dict, bundle: dict, allocator_result: dict, job: Path, store: Path = MODEL_ROOT) -> dict:
    """Retain returned USDs only after provider closeout and independent byte verification."""
    from .decision_evidence_contracts import canonical_digest
    from .deployment_embodiment_runtime_configuration import verify_runtime_configuration, bind_prepared_configuration
    from .deployment_embodiment_asset_import import verify_prepared_asset_directory
    from .immutable_directory_publication import publish_staged_immutable_directory

    require(allocator_result.get("status") == "completed"
            and allocator_result.get("bundle_sha256") == bundle["bundle_sha256"]
            and allocator_result.get("continuing_spend_from_this_run") is False,
            "provider_closeout_pending")
    receipt_path = Path(allocator_result["native_control_result_path"])
    require(receipt_path.is_relative_to(workspace_for(job) / "allocator") and receipt_path.name == RESULT_FILENAME,
            "result_path_invalid")
    result = validate_preparation_result(read(receipt_path, limit=32 * 1024**2),
        expected_inputs_digest=bundle["inputs_digest"], expected_ids=request["embodiment_ids"], output_root=receipt_path.parent)
    require(result["result_digest"] == allocator_result.get("native_control_result_digest"), "result_binding_changed")
    source_rows = {row["embodiment_id"]: row for row in bundle["inputs"]["sources"]}
    store.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(not any(p.is_symlink() for p in (store, *store.parents)), "model_store_path_invalid")
    rows = []
    for row in result["results"]:
        identity = row["embodiment_id"]
        if row["status"] != "imported":
            rows.append(row)
            continue
        try:
            body = receipt_path.parent / "bodies" / identity
            configuration = verify_runtime_configuration(body / "configuration")
            imported = verify_prepared_asset_directory(body / "prepared")
            require(configuration["source_bundle_digest"] == source_rows[identity]["bundle_digest"]
                    and row.get("configuration_digest") == configuration["configuration_digest"]
                    and row.get("import_digest") == imported["import_digest"]
                    and row.get("model_sha256") == imported["model"]["sha256"], "returned_body_identity_changed")
            raw_binding = read(body / "binding-inputs.json")
            binding = bind_prepared_configuration(body / "configuration", body / "prepared",
                                                  raw_binding["robot_embodiment"]["model"]["url"])
            require(binding == raw_binding, "returned_binding_changed")
            version = canonical_digest({"configuration_digest": configuration["configuration_digest"],
                                        "model_sha256": imported["model"]["sha256"]})[7:]
            destination = store / identity / version
            destination.parent.mkdir(exist_ok=True)
            if destination.exists():
                retained = bind_prepared_configuration(destination / "configuration", destination / "prepared",
                                                      binding["robot_embodiment"]["model"]["url"])
                require(retained == binding, "retained_model_changed")
            else:
                with tempfile.TemporaryDirectory(prefix=".asset-promote-", dir=destination.parent) as temporary:
                    staged = Path(temporary)
                    shutil.copytree(body, staged, dirs_exist_ok=True)
                    publish_staged_immutable_directory(staging=staged, destination=destination,
                        manifest_name="body-result.json", output_exists_code="embodiment_preparation_model_exists")
            try:
                diagnostic = retain_batch_diagnostic(body=body, row=row, binding=binding, request=request, store=store)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                diagnostic = {"status": "failed", "verification_status": "unverified", "smoke_digest": None,
                    "process_digest": None, "candidate_policy_queried": False, "task_executed": False,
                    "physical_success_proven": False, "claim_ceiling": "native_controller_diagnostic_only",
                    "reasons": ["diagnostic_retention_failed:" + str(exc)[:240]]}
            rows.append({**row, "native_controller_smoke": diagnostic,
                         "status": "retained", "directory": str(destination / "prepared"),
                         "binding_inputs_path": str(destination / "binding-inputs.json"),
                         "binding_inputs_sha256": sha(destination / "binding-inputs.json"),
                         "model_url": binding["robot_embodiment"]["model"]["url"],
                         "model_delivery": "operator_prepared_directory", "public_download_published": False})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rows.append({"embodiment_id": identity, "status": "import_verification_failed", "reason": str(exc)[:500]})
    summary = sealed({"schema_version": "embodiment_asset_preparation_publication.v1", "status": "completed",
        "claim_ceiling": "robot_asset_import_only", "request_digest": request["request_digest"],
        "bundle_sha256": bundle["bundle_sha256"], "result_digest": result["result_digest"], "results": rows,
        "candidate_policy_queried": False, "task_executed": False}, "publication_digest")
    write_once(store / "batches" / (request["authorization_id"] + ".json"), summary)
    write_once(job / "publication.json", summary)
    return summary


def unavailable_rows(request: dict, reason: str, details=None) -> list[dict]:
    return [{"embodiment_id": identity, "status": "not_imported", "reason": reason,
             **({"details": details} if details else {})} for identity in request["embodiment_ids"]]


def process_reserved_entry(item: Path, *, root: Path, store: Path, commit: str) -> dict:
    from .control_plane_disk_budget import reserve_control_plane_disk, DEFAULT_RESERVATION_ROOT
    from .native_task_runtime_source_packet import verify_native_task_runtime_source_packet
    request = read(root / "authorizations" / item.name)
    job = root / "jobs" / request["authorization_id"]
    workspace = workspace_for(job)
    if (job / "publication.json").exists():
        return read(job / "publication.json")
    resumed = (job / "allocator_launch_intent.json").exists() or (job / "allocator_result.json").exists()
    runtime = None if resumed else runtime_source_receipt()
    packet = None if resumed else verify_native_task_runtime_source_packet(runtime)
    # The bundle carries the native packet streamed from its verified path, plus
    # source cache/staging and program overhead. The returned ZIP and its
    # extraction each take their own exact reservation when they arrive. Holding
    # them here as well counted this job's own hold against its download, so even
    # a 50 KB result was refused once the bundle was written
    # (public-robot-imports-20261030-05).
    peak = (MAX_OUTPUT_BYTES if resumed else
            packet["packet_size_bytes"] + 4 * 1024**3)
    ledger = os.environ.get("BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT") or DEFAULT_RESERVATION_ROOT
    with reserve_control_plane_disk("embodiment_preparation", target_root=workspace, expected_bytes=peak,
            reservation_root=ledger, workspace=workspace, ttl_seconds=6 * 3600) as work_hold:
        with reserve_control_plane_disk("embodiment_preparation", target_root=store, expected_bytes=MAX_OUTPUT_BYTES,
                reservation_root=ledger, ttl_seconds=6 * 3600) as model_hold:
            job.mkdir(parents=True, exist_ok=True)
            if not (job / "disk-admission.json").exists():
                write_once(job / "disk-admission.json", {"workspace": work_hold.receipt(), "models": model_hold.receipt(),
                    "workspace_path": str(workspace), "authorization_id": request["authorization_id"]})
            # The allocator inherits this; its download and extraction write
            # inside the workspace hold rather than beside it.
            from .control_plane_disk_budget import ENCLOSING_RESERVATIONS_ENV
            previous = os.environ.get(ENCLOSING_RESERVATIONS_ENV)
            token = work_hold.receipt().get("reservation_token")
            if isinstance(token, str):
                os.environ[ENCLOSING_RESERVATIONS_ENV] = ",".join(filter(None, (previous, token)))
            try:
                return process_entry(item, root=root, store=store, commit=commit, resolve_runtime=lambda: runtime)
            finally:
                if previous is None:
                    os.environ.pop(ENCLOSING_RESERVATIONS_ENV, None)
                else:
                    os.environ[ENCLOSING_RESERVATIONS_ENV] = previous


def process_entry(item: Path, *, root: Path, store: Path, commit: str, build=build_bundle,
                  runner=None, resolve_runtime=runtime_source_receipt) -> dict:
    from .rigid_robot_free_probe_runner import run_allocator, _allocator_alive
    entry = read(item)
    request = validate_request(read(root / "authorizations" / (entry["authorization_id"] + ".json")), allow_expired=True)
    require(entry["request_digest"] == request["request_digest"], "queue_binding_changed")
    job = root / "jobs" / request["authorization_id"]
    job.mkdir(parents=True, exist_ok=True)
    if (job / "publication.json").exists():
        return read(job / "publication.json")
    result_path = job / "allocator_result.json"
    retained_result = workspace_for(job) / "allocator/adp_arena_vast_result.json"
    if not result_path.exists() and (job / "allocator_launch_intent.json").exists() and retained_result.is_file():
        # The wrapper may have exited after transport durably closed its attempt.
        # Adopt only its exact bound terminal receipt; never create another GPU.
        recovered = read(retained_result, limit=32 * 1024**2)
        intent = read(job / "allocator_launch_intent.json")
        require(recovered.get("bundle_sha256") == intent["bundle_sha256"], "recovered_bundle_changed")
        if recovered.get("status") == "completed" and recovered.get("continuing_spend_from_this_run") is False:
            write_once(result_path, recovered)
    if not result_path.exists() and (job / "allocator_launch_intent.json").exists():
        # A restarted controller never calls create again. A live child is awaited;
        # unknown outcomes stay visible for retained result/teardown reconciliation.
        return {"status": "running" if _allocator_alive(job) else "reconciliation_required",
                "authorization_id": request["authorization_id"], "workspace_path": str(workspace_for(job)), "provider_relaunch_performed": False,
                "results": unavailable_rows(request, "allocation_outcome_pending_reconciliation")}
    if not result_path.exists():
        validate_request(request)
        require(request["release_commit"] == commit, "installed_release_mismatch")
        bundle = build(request=request, runtime_source_receipt=resolve_runtime(),
                       source_root=workspace_for(job).parent / "sources" / commit, output=workspace_for(job) / "bundle")
        write_once(job / "allocator_launch_intent.json", {"request_digest": request["request_digest"],
            "bundle_sha256": bundle["bundle_sha256"], "release_commit": commit})
        (runner or run_allocator)(allocator_argv(request, job), attempt=job)
    if not result_path.exists():
        return {"status": "reconciliation_required", "authorization_id": request["authorization_id"],
                "results": unavailable_rows(request, "allocation_outcome_pending_reconciliation")}
    result = read(result_path, limit=32 * 1024**2)
    if result.get("status") != "completed":
        return {"status": "not_completed", "authorization_id": request["authorization_id"],
                "blockers": result.get("blockers", []), "continuing_spend_from_this_run": result.get("continuing_spend_from_this_run"),
                "provider_relaunch_performed": False,
                "results": unavailable_rows(request, "allocation_not_completed", result.get("blockers", []))}
    # Post-return promotion can resume after expiry, but never under different code.
    require(request["release_commit"] == commit, "retained_release_required_for_promotion")
    bundle = load_verified_bundle(workspace_for(job) / "bundle" / RECEIPT, expected_implementation_commit=commit, allow_expired=True)
    return promote(request=request, bundle=bundle, allocator_result=result, job=job, store=store)


BUNDLE_ARCHIVE = "embodiment_asset_preparation.zip"
WORKSPACE_RELEASE = "workspace-release.json"


def _terminal(job: Path) -> bool:
    """Nothing reads this job's upload again: it is published, or its one GPU attempt ended unpromotable."""
    if (job / "publication.json").exists():
        return True
    result = job / "allocator_result.json"
    return result.is_file() and read(result, limit=32 * 1024**2).get("status") != "completed"


def release_terminal_workspaces(root: Path, *, commit: str) -> list[dict]:
    """Free each terminal batch's uploaded bundle archive and other releases' source caches.

    The archive (the 4.5 GB runtime packet plus sources) is only the GPU upload: a
    published or unpromotable batch never reads it again, and a batch is never
    relaunched. Its receipt, the job state and the model store are kept; the archive's
    digest is recorded in the job's ``workspace-release.json``. Source caches of
    other releases are verified public downloads a later build re-materialises.
    Twelve batches had kept 53 GB on the shared work disk (2026-10-31).
    """
    released = []
    for job in sorted(path for path in (root / "jobs").glob("*") if path.is_dir()):
        archive = workspace_for(job) / "bundle" / BUNDLE_ARCHIVE
        try:
            if archive.is_symlink() or not archive.is_file() or not _terminal(job):
                continue
            receipt = read(archive.parent / RECEIPT, limit=32 * 1024**2)
            record = {"schema_version": "embodiment_preparation_workspace_release.v1", "authorization_id": job.name,
                      "bundle_path": str(archive), "bundle_sha256": receipt.get("bundle_sha256"),
                      "released_bytes": archive.stat().st_size, "model_store_changed": False}
            if not (job / WORKSPACE_RELEASE).exists():
                write_once(job / WORKSPACE_RELEASE, record)
            # The bundle was published read-only (0555); open it to its owner only to unlink.
            mode = archive.parent.stat().st_mode & 0o777
            archive.parent.chmod(mode | 0o700)
            try:
                archive.unlink()
            finally:
                archive.parent.chmod(mode)
            released.append(record)
        except (OSError, ValueError) as exc:
            # Freeing space never stops the queue; the reason is reported and retried next pass.
            released.append({"authorization_id": job.name, "release_blocked": type(exc).__name__})
    sources = (WORK_ROOT if root == STATE_ROOT else root / "work") / "sources"
    for cache in sorted(path for path in sources.glob("*") if path.is_dir() and not path.is_symlink()):
        if cache.name == commit:
            continue
        try:
            for directory in (cache, *(p for p in cache.rglob("*") if p.is_dir() and not p.is_symlink())):
                directory.chmod(0o700)
            shutil.rmtree(cache)
            released.append({"source_cache_release": cache.name})
        except OSError as exc:
            released.append({"source_cache_release": cache.name, "release_blocked": type(exc).__name__})
    return released


def run_queue(*, root: Path = STATE_ROOT, store: Path = MODEL_ROOT) -> dict:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / ".controller.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "already_running"}
        released = release_terminal_workspaces(root, commit=source_commit())
        outcomes = []
        for item in sorted((root / "pending").glob("*.json")):
            try:
                outcome = process_reserved_entry(item, root=root, store=store, commit=source_commit())
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                outcome = {"status": "blocked", "authorization_id": item.stem, "reason": str(exc)[:500]}
                try:
                    request = read(root / "authorizations" / item.name)
                    outcome["results"] = unavailable_rows(request, str(exc)[:500])
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            outcomes.append(outcome)
            if outcome["status"] == "completed":
                (root / "completed").mkdir(exist_ok=True)
                os.replace(item, root / "completed" / item.name)
            # One operator grant per service pass; each already bound to <=one GPU.
            if outcome["status"] not in {"completed", "not_completed", "blocked"}:
                break
        released += release_terminal_workspaces(root, commit=source_commit())
        summary = {"status": "checked", "results": outcomes, "workspace_releases": released}
        path = root / "latest.json"
        temporary = root / ".latest.json"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(temporary, path)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("enqueue", "run-queue"))
    parser.add_argument("--request-json")
    args = parser.parse_args(argv)
    result = enqueue(json.loads(args.request_json)) if args.command == "enqueue" else run_queue()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
