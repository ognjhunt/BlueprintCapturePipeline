"""Derive immutable per-release scene machinery from actual deployment receipts."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_submission_inputs import read, checked_file, release_inputs
from .task_evaluation_scene_progression_state import require, safe_path
from .task_evaluation_public_scene_attempt_factory import record, RELEASE_SCHEMA
from .task_evaluation_scene_intake import write_exclusive


def resolve_release_binding(config, *, running_commit):
    """No copied commit strings: prove deployed surfaces and runtime publications.

    A deployment may finish between timer ticks. Until its final receipt exists,
    progression refuses rather than pairing new source with old authorization.
    """
    if not config.get("deployment_receipt_root"):
        value = read(safe_path(config["release_binding_path"]), digest_field="release_digest")
        require(value.get("source_commit") == running_commit, "running_release_mismatch")
        return value
    root = safe_path(config["release_binding_root"]) / running_commit
    path = root / "release.json"
    if path.exists():
        value = read(path, digest_field="release_digest")
        require(value.get("source_commit") == running_commit, "running_release_mismatch")
        for key in ("deploy_receipt", "release_provenance", "release_environment"):
            checked_file(value[key]["path"], value[key])
        release_inputs(deploy_path=Path(value["deploy_receipt"]["path"]),
            provenance_path=Path(value["release_provenance"]["path"]),
            publication_root=Path(value["runtime_publication_root"]), commit=running_commit,
            release_admission_mode=value["release_admission_mode"])
        return value
    candidates = []
    for candidate in safe_path(config["deployment_receipt_root"]).glob("*.json"):
        deploy = read(candidate)
        if deploy.get("source_commit") == running_commit and deploy.get("status") == "deployed":
            candidates.append((candidate, deploy))
    require(bool(candidates), "current_deployment_receipt_missing")
    # Several verified deployments of identical source can be retained. Bind one
    # exact receipt once; later administrative receipt writes cannot rebind it.
    candidate, deploy = sorted(candidates, key=lambda row: row[0].name)[0]
    repo = safe_path(deploy["release_path"])
    require(repo == Path(config["running_repo_root"]).resolve(strict=True), "deployed_repo_mismatch")
    provenance_ref = deploy["release_provenance"]
    provenance = checked_file(provenance_ref["path"], provenance_ref)
    env_ref = deploy["scene_configuration_environment"]
    env = checked_file(env_ref["path"], env_ref)
    require(deploy["scene_configuration_environment"].get("credential_values_recorded") is False,
            "release_environment_secret_boundary_missing")
    mode = "development_iteration" if provenance_ref.get("provenance_status") == "iteration" else "promoted"
    _, toolchain, renderer = release_inputs(deploy_path=candidate, provenance_path=provenance,
        publication_root=safe_path(config["runtime_publication_root"]), commit=running_commit,
        release_admission_mode=mode)
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    env_snapshot = root / "release.env"
    payload = env.read_bytes()
    if not env_snapshot.exists():
        descriptor = os.open(env_snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o440)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    require(env_snapshot.read_bytes() == payload, "release_environment_snapshot_conflict")
    value = {"schema_version": RELEASE_SCHEMA, "source_commit": running_commit,
        "runtime_digest": canonical_digest({"toolchain": toolchain, "renderer": renderer}),
        "repo_root": str(repo), "runtime_publication_root": str(safe_path(config["runtime_publication_root"])),
        "namespace_timestamp": datetime.fromtimestamp(candidate.stat().st_mtime, timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "release_admission_mode": mode, "deploy_receipt": record(candidate),
        "release_provenance": record(provenance), "release_environment": record(env_snapshot)}
    value["release_digest"] = canonical_digest(value, digest_field="release_digest")
    try:
        write_exclusive(path, value)
    except FileExistsError:
        require(read(path, digest_field="release_digest") == value, "release_binding_conflict")
    return value
