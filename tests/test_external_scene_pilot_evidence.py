from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.external_scene_pilot_evidence import (
    MANDATORY_GATE_IDS,
    ExternalScenePilotEvidenceError,
    compile_external_scene_pilot_evidence,
    write_external_scene_pilot_evidence,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _git_repo(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    (root / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=root, check=True)
    # The compiler records upstream divergence, so give the fixture a local
    # upstream without requiring network access.
    subprocess.run(["git", "branch", "upstream"], cwd=root, check=True)
    subprocess.run(
        ["git", "branch", "--set-upstream-to=upstream", "main"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    return root, commit


def _profile(asset_digest: str, source_class: str) -> dict[str, object]:
    return {
        "asset_digest": asset_digest,
        "source_class": source_class,
        "capture_modality": "unknown",
        "archive_structure": "usdz",
        "nurec_representation": "provider_nurec",
        "mesh_collider_presence": True,
        "metric_scale": "provider_declared_not_independently_validated",
        "up_axis": "Z",
        "bounds": [1.0, 2.0, 3.0],
        "unresolved_references": [],
        "runtime_cost_usd": 0.5,
        "manual_operator_minutes": 4,
        "failure_points": [],
    }


def _request(
    *, commit: str, artifact: Path, artifact_root: Path, incomplete_gate: str | None = None
) -> dict[str, object]:
    reference = {
        "path": artifact.relative_to(artifact_root).as_posix(),
        "sha256": _sha256(artifact),
    }
    lanes = []
    for lane_id, source_class in (
        ("public_reference", "public_provider_sample"),
        ("private_site", "user_managed_provider_export"),
    ):
        gates = []
        for gate_id in MANDATORY_GATE_IDS:
            status = "not_run" if gate_id == incomplete_gate else "passed"
            gates.append(
                {
                    "gate_id": gate_id,
                    "status": status,
                    "blocker_codes": (["runtime_not_run"] if status == "not_run" else []),
                    "artifacts": ([] if status == "not_run" else [reference]),
                }
            )
        lanes.append(
            {
                "lane_id": lane_id,
                "remote_runtime_used": True,
                "comparison_profile": _profile(_sha256(artifact), source_class),
                "gates": gates,
            }
        )
    claim_status = "abstained" if incomplete_gate else "supported"
    return {
        "schema_version": "external_scene_pilot_finalization_request.v1",
        "run_id": "niantic-isaac-pilot-test",
        "source_commit_sha": commit,
        "lanes": lanes,
        "claims": [
            {
                "claim_id": "downstream-harness",
                "lane_id": "public_reference",
                "statement": "Exact scene evidence reached the evaluation harness.",
                "status": claim_status,
                "required_gate_ids": [
                    "isaac_stage_load",
                    "nonblank_render",
                    "collision_contact",
                    "two_candidate_traces",
                    "distinct_trace_pair",
                    "task_evaluation_run",
                ],
                "blocker_codes": (["runtime_not_run"] if incomplete_gate else []),
            }
        ],
        "reproduction_commands": ["python -m blueprint_pipeline.example --request request.json"],
    }


def test_compiles_complete_dual_lane_reports(tmp_path: Path) -> None:
    repo, commit = _git_repo(tmp_path)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact = artifact_root / "runtime.json"
    artifact.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    request = _request(commit=commit, artifact=artifact, artifact_root=artifact_root)

    reports = compile_external_scene_pilot_evidence(
        request, artifact_root=artifact_root, repo_root=repo
    )

    assert reports["terminal_summary"]["status"] == "dual_asset_harness_complete"
    assert reports["terminal_summary"]["scientific_thesis_state"] == "inconclusive"
    assert all(
        lane["completion_status"] == "complete" for lane in reports["master_manifest"]["lanes"]
    )
    assert reports["claim_ledger"]["claims"][0]["status"] == "supported"
    assert reports["git_state"]["clean"] is True
    assert reports["master_manifest"]["master_manifest_digest"].startswith("sha256:")

    written = write_external_scene_pilot_evidence(reports, tmp_path / "reports")
    assert set(written) == {
        "master_manifest",
        "comparison",
        "claim_ledger",
        "terminal_summary",
        "reproduction",
        "git_state",
    }
    assert all(Path(path).is_file() for path in written.values())


def test_live_gate_not_run_keeps_pilot_partial(tmp_path: Path) -> None:
    repo, commit = _git_repo(tmp_path)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact = artifact_root / "runtime.json"
    artifact.write_text("{}", encoding="utf-8")
    request = _request(
        commit=commit,
        artifact=artifact,
        artifact_root=artifact_root,
        incomplete_gate="distinct_trace_pair",
    )

    reports = compile_external_scene_pilot_evidence(
        request, artifact_root=artifact_root, repo_root=repo
    )

    assert reports["terminal_summary"]["status"] == "partial_harness_evidence"
    assert reports["terminal_summary"]["remaining_gate_ids"] == {
        "private_site": ["distinct_trace_pair"],
        "public_reference": ["distinct_trace_pair"],
    }


def test_detached_head_uses_origin_main_for_divergence(tmp_path: Path) -> None:
    repo, commit = _git_repo(tmp_path)
    subprocess.run(["git", "update-ref", "refs/remotes/origin/main", commit], cwd=repo, check=True)
    subprocess.run(["git", "switch", "--detach", commit], cwd=repo, check=True)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact = artifact_root / "runtime.json"
    artifact.write_text("{}", encoding="utf-8")
    request = _request(commit=commit, artifact=artifact, artifact_root=artifact_root)

    reports = compile_external_scene_pilot_evidence(
        request, artifact_root=artifact_root, repo_root=repo
    )

    assert reports["git_state"]["branch"] is None
    assert reports["git_state"]["upstream"] is None
    assert reports["git_state"]["comparison_ref"] == "origin/main"
    assert reports["git_state"]["ahead"] == 0
    assert reports["git_state"]["behind"] == 0


def test_cpu_and_task_evaluation_abstentions_are_completion_safe(tmp_path: Path) -> None:
    repo, commit = _git_repo(tmp_path)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact = artifact_root / "runtime.json"
    artifact.write_text("{}", encoding="utf-8")
    request = _request(commit=commit, artifact=artifact, artifact_root=artifact_root)
    for lane in request["lanes"]:
        for gate in lane["gates"]:
            if gate["gate_id"] in {"cpu_qualification", "task_evaluation_run"}:
                gate["status"] = "abstained"
                gate["blocker_codes"] = ["formal_claim_inputs_missing"]
    request["claims"][0]["status"] = "abstained"
    request["claims"][0]["blocker_codes"] = ["formal_claim_inputs_missing"]

    reports = compile_external_scene_pilot_evidence(
        request, artifact_root=artifact_root, repo_root=repo
    )

    assert reports["terminal_summary"]["status"] == "dual_asset_harness_complete"


def test_rejects_digest_mismatch(tmp_path: Path) -> None:
    repo, commit = _git_repo(tmp_path)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact = artifact_root / "runtime.json"
    artifact.write_text("{}", encoding="utf-8")
    request = _request(commit=commit, artifact=artifact, artifact_root=artifact_root)
    request["lanes"][0]["gates"][0]["artifacts"][0]["sha256"] = "sha256:" + "0" * 64

    with pytest.raises(ExternalScenePilotEvidenceError) as raised:
        compile_external_scene_pilot_evidence(request, artifact_root=artifact_root, repo_root=repo)

    assert any(code.startswith("artifact_digest_mismatch:") for code in raised.value.codes)


def test_rejects_supported_claim_with_unqualified_gate(tmp_path: Path) -> None:
    repo, commit = _git_repo(tmp_path)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact = artifact_root / "runtime.json"
    artifact.write_text("{}", encoding="utf-8")
    request = _request(
        commit=commit,
        artifact=artifact,
        artifact_root=artifact_root,
        incomplete_gate="two_candidate_traces",
    )
    request["claims"][0]["status"] = "supported"

    with pytest.raises(ExternalScenePilotEvidenceError) as raised:
        compile_external_scene_pilot_evidence(request, artifact_root=artifact_root, repo_root=repo)

    assert "pilot_claim_support_unqualified:downstream-harness" in raised.value.codes
