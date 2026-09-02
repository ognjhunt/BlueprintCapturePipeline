from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from blueprint_pipeline import live_pipeline_intake_service as service
from blueprint_pipeline.live_pipeline_result_artifact_resolution import (
    resolve_live_pipeline_result_artifact,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.live_pipeline_control_plane import (
    CONTROL_PLANE_OUTPUT_PATH_ENV,
)


ARTIFACT_ID = "a2634d9fa2634d9fa2634d9fa2634d9f"
RESULT_ROOT = "/var/lib/blueprint/pipeline-control-plane/task-evaluation-policy-canaries"


def test_intake_unit_and_environment_install_explicit_canary_result_root() -> None:
    repository = Path(__file__).resolve().parents[1]
    binding = (
        "BLUEPRINT_TASK_EVALUATION_POLICY_CANARY_RESULT_ROOT=" f"{RESULT_ROOT}"
    )
    unit = (
        repository / "deploy/systemd/blueprint-pipeline-intake.service"
    ).read_text(encoding="utf-8")
    environment = (
        repository / "deploy/systemd/pipeline-control-plane.env.example"
    ).read_text(encoding="utf-8")

    assert f"Environment={binding}" in unit
    assert binding in environment


def _configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, str, Path]:
    manifest = tmp_path / "control" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    token = "artifact-route-test-secret"
    canary_root = tmp_path / "policy-canaries"
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest))
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV, str(tmp_path / "work"))
    monkeypatch.setenv(
        service.INTAKE_CLIENT_SECRETS_ENV,
        json.dumps({"blueprint-webapp": token}),
    )
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / "nonces"))
    monkeypatch.setenv(
        service.TASK_EVALUATION_POLICY_CANARY_RESULT_ROOT_ENV,
        str(canary_root),
    )
    monkeypatch.delenv(service.INTAKE_TOKEN_ENV, raising=False)
    monkeypatch.delenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, raising=False)
    return TestClient(service.create_app()), token, canary_root


def _headers(token: str, nonce: str) -> dict[str, str]:
    timestamp = datetime.now(timezone.utc).isoformat()
    signature = hmac.new(
        token.encode("utf-8"),
        f"{timestamp}.blueprint-webapp.{nonce}.".encode("utf-8"),
        "sha256",
    ).hexdigest()
    return {
        "x-blueprint-pipeline-timestamp": timestamp,
        "x-blueprint-pipeline-client-id": "blueprint-webapp",
        "x-blueprint-pipeline-nonce": nonce,
        "x-blueprint-pipeline-signature": f"sha256={signature}",
    }


def _registry(
    run_root: Path,
    *,
    run_id: str,
    content: bytes = b"review-video",
    evidence_root: Path | None = None,
) -> None:
    evidence = evidence_root or run_root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    artifact = evidence / "external.mp4"
    artifact.write_bytes(content)
    record = {
        "artifact_id": ARTIFACT_ID,
        "role": "review_video",
        "relative_path": artifact.relative_to(evidence).as_posix(),
        "sha256": "sha256:" + hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "content_type": "video/mp4",
        "evidence_root": str(evidence),
    }
    value = {
        "schema_version": "task_evaluation_result_artifact_registry.v1",
        "run_id": run_id,
        "delivery_digest": "sha256:" + "d" * 64,
        "artifacts": [record],
        "registry_digest": "",
    }
    value["registry_digest"] = canonical_digest(
        value, digest_field="registry_digest"
    )
    registry = run_root / "artifacts" / "result_delivery" / "artifact_registry.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps(value), encoding="utf-8")


def _get(client: TestClient, token: str, run_id: str, nonce: str):
    return client.get(
        f"/api/live-pipeline/task-evaluation-runs/{run_id}/artifacts/{ARTIFACT_ID}",
        headers=_headers(token, nonce),
    )


def test_canary_artifact_resolves_from_exact_activation_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, token, canary_root = _configure(tmp_path, monkeypatch)
    run_id = "scene839873-policy-canary-f23e2100"
    _registry(canary_root / f"{run_id}-activation", run_id=run_id)

    response = _get(client, token, run_id, "canary-video")

    assert response.status_code == 200
    assert response.content == b"review-video"
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["content-disposition"].startswith("inline;")


def test_unknown_canary_run_and_artifact_remain_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, token, canary_root = _configure(tmp_path, monkeypatch)
    run_id = "scene839873-policy-canary-known"
    _registry(canary_root / f"{run_id}-activation", run_id=run_id)

    assert _get(client, token, "unknown-run", "unknown-run").status_code == 404
    response = client.get(
        f"/api/live-pipeline/task-evaluation-runs/{run_id}/artifacts/"
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        headers=_headers(token, "unknown-artifact"),
    )
    assert response.status_code == 404


def test_canary_resolution_rejects_cross_run_traversal_and_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, token, canary_root = _configure(tmp_path, monkeypatch)
    run_id = "scene839873-policy-canary-requested"
    _registry(canary_root / f"{run_id}-activation", run_id="different-run")
    assert _get(client, token, run_id, "cross-run").status_code == 404

    with pytest.raises(ValueError, match="run_id must match"):
        resolve_live_pipeline_result_artifact(
            legacy_state_root=Path("/tmp/task-evaluation-runs"),
            policy_canary_result_root=canary_root,
            run_id="../different-run",
            artifact_id=ARTIFACT_ID,
        )

    linked_run = "scene839873-policy-canary-linked"
    target = canary_root / "unrelated-activation"
    _registry(target, run_id=linked_run)
    (canary_root / f"{linked_run}-activation").symlink_to(
        target, target_is_directory=True
    )
    assert _get(client, token, linked_run, "symlink-run").status_code == 404


def test_legacy_v2_v3_run_root_still_wins_over_canary_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, token, canary_root = _configure(tmp_path, monkeypatch)
    run_id = "legacy-task-evaluation-run"
    legacy_root = tmp_path / "work" / "task_evaluation_runs" / "runs" / run_id
    _registry(legacy_root, run_id=run_id, content=b"legacy-video")
    _registry(
        canary_root / f"{run_id}-activation",
        run_id=run_id,
        content=b"wrong-canary-video",
    )

    response = _get(client, token, run_id, "legacy-run")

    assert response.status_code == 200
    assert response.content == b"legacy-video"


def test_registry_cannot_point_to_another_runs_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, token, canary_root = _configure(tmp_path, monkeypatch)
    run_id = "scene839873-policy-canary-contained"
    run_root = canary_root / f"{run_id}-activation"
    other_evidence = canary_root / "other-run-activation" / "evidence"
    _registry(run_root, run_id=run_id, evidence_root=other_evidence)

    assert _get(client, token, run_id, "outside-evidence").status_code == 404
