from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.dual_task_evidence_package_index import (
    HTML_FILENAME,
    PACKAGE_INDEX_FILENAME,
    DualTaskEvidencePackageIndexError,
    materialize_dual_task_evidence_package_index,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "docs/arm_decision_proof_v1/third_scene_dual_task_evidence"


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _task_index(package: Path, task_id: str) -> None:
    task_dir = package / task_id
    payload: dict[str, object] = {
        "schema_version": "adp_manipulation_episode_evidence_index.v1",
        "run_identity": {
            "scene_id": "fixture_scene",
            "task_id": task_id,
            "scenario_suite_digest": "sha256:" + "1" * 64,
        },
        "episodes": [],
        "episode_count": 0,
        "required_camera_ids": ["external", "wrist", "overview"],
        "overview_is_review_only": True,
        "scores_are_deterministic_simulator_state": True,
        "review_videos_are_not_physical_truth": True,
        "typed_abstention": {"status": "typed_evidence_backed_abstention"},
        "supporting_evidence": [],
        "index_digest": "",
    }
    payload["index_digest"] = canonical_digest(payload, digest_field="index_digest")
    _write_json(task_dir / "episode_evidence_index.v1.json", payload)
    (task_dir / "OPEN_ME_episode_evidence_index.html").write_text(
        "<!doctype html><p>task</p>\n", encoding="utf-8"
    )


def _shared_manifest(workspace: Path, relative_path: str) -> None:
    payload: dict[str, object] = {
        "schema_version": "third_scene_cad_agent_visual_comparison_binding.v1",
        "status": "digest_bound_private_review_media_available",
        "binding_digest": "",
    }
    payload["binding_digest"] = canonical_digest(
        payload, digest_field="binding_digest"
    )
    _write_json(workspace / relative_path, payload)


def test_package_index_materializes_from_verified_tasks_and_manifests(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    package = workspace / "docs/evidence"
    package.mkdir(parents=True)
    _task_index(package, "task_a")
    _task_index(package, "task_b")
    _shared_manifest(workspace, "docs/manifests/cad_visual.json")

    receipt = materialize_dual_task_evidence_package_index(
        workspace_root=workspace,
        package_root=package,
        title="Fixture package",
        status_summary="typed abstention",
        blocker_summary="external blocker",
        cad_summary="CAD review media bound",
        task_indexes=[
            {
                "label": "Task A",
                "task_id": "task_a",
                "relative_path": "task_a/episode_evidence_index.v1.json",
            },
            {
                "label": "Task B",
                "task_id": "task_b",
                "relative_path": "task_b/episode_evidence_index.v1.json",
            },
        ],
        shared_manifests=[
            {
                "label": "CAD visual comparison binding",
                "relative_path": "docs/manifests/cad_visual.json",
            }
        ],
    )

    assert receipt["schema_version"] == "adp_dual_task_evidence_package_index.v1"
    assert receipt["task_count"] == 2
    assert receipt["shared_manifest_count"] == 1
    assert receipt["package_index_digest"].startswith("sha256:")
    assert (package / PACKAGE_INDEX_FILENAME).is_file()
    assert (package / HTML_FILENAME).is_file()


def test_package_index_rejects_tampered_task_index(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    package = workspace / "docs/evidence"
    package.mkdir(parents=True)
    _task_index(package, "task_a")
    _shared_manifest(workspace, "docs/manifests/cad_visual.json")
    task_path = package / "task_a/episode_evidence_index.v1.json"
    payload = json.loads(task_path.read_text())
    payload["episode_count"] = 99
    task_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        DualTaskEvidencePackageIndexError,
        match="evidence_package_task_index_digest_invalid",
    ):
        materialize_dual_task_evidence_package_index(
            workspace_root=workspace,
            package_root=package,
            title="Fixture package",
            status_summary="typed abstention",
            blocker_summary="external blocker",
            cad_summary="CAD review media bound",
            task_indexes=[
                {
                    "label": "Task A",
                    "task_id": "task_a",
                    "relative_path": "task_a/episode_evidence_index.v1.json",
                }
            ],
            shared_manifests=[
                {
                    "label": "CAD visual comparison binding",
                    "relative_path": "docs/manifests/cad_visual.json",
                }
            ],
        )


def test_checked_in_package_index_exposes_cad_visual_comparison_binding() -> None:
    html = (PACKAGE / "index.html").read_text(encoding="utf-8")
    receipt = json.loads((PACKAGE / PACKAGE_INDEX_FILENAME).read_text(encoding="utf-8"))

    assert (
        "third_scene_840920_dual_task_cad_agent_visual_comparison_binding.v1.json"
        in html
    )
    assert receipt["package_index_digest"] == canonical_digest(
        receipt, digest_field="package_index_digest"
    )
    assert any(
        row["schema_version"] == "third_scene_cad_agent_visual_comparison_binding.v1"
        and row["receipt_digest"]
        == "sha256:b41c3adfedaf585dd7e9031e7cfbc7fd2847e03df546d544382402bde6c81fbe"
        for row in receipt["shared_manifests"]
    )

