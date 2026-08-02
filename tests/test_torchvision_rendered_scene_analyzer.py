from __future__ import annotations

import hashlib
from pathlib import Path

import blueprint_pipeline.torchvision_rendered_scene_analyzer as analyzer


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _payload(tmp_path: Path, *, robot_id: str = "franka_panda") -> tuple[dict, Path]:
    rgb = tmp_path / "view.png"
    rgb.write_bytes(b"exact-rendered-rgb")
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"pinned-weights")
    payload = {
        "analyzer_request": {
            "analyzer_request_digest": "sha256:" + "a" * 64,
            "robot_id": robot_id,
            "task_context": {},
            "rendered_views": [
                {
                    "view_id": "third_person",
                    "rgb_digest": _digest(b"exact-rendered-rgb"),
                }
            ],
        },
        "runtime_inputs": {
            "rendered_views": [
                {
                    "view_id": "third_person",
                    "rgb_path": str(rgb),
                }
            ]
        },
    }
    return payload, weights


def test_local_analyzer_proposes_taskable_franka_objects(tmp_path, monkeypatch) -> None:
    payload, weights = _payload(tmp_path)
    monkeypatch.setattr(
        analyzer, "MODEL_WEIGHT_SHA256", hashlib.sha256(b"pinned-weights").hexdigest()
    )

    result = analyzer.analyze_payload(
        payload,
        weights_path=weights,
        detector=lambda _paths: [
            [
                {
                    "label": "chair",
                    "score": 0.99,
                    "bbox_xyxy_pixels": [1, 2, 30, 40],
                },
                {
                    "label": "sink",
                    "score": 0.88,
                    "bbox_xyxy_pixels": [100, 200, 500, 600],
                },
                {
                    "label": "bottle",
                    "score": 0.95,
                    "bbox_xyxy_pixels": [40, 50, 70, 120],
                },
            ]
        ],
    )

    assert result["status"] == "completed"
    assert result["candidate_may_self_authorize"] is False
    assert [row["object_label"] for row in result["proposals"]] == ["bottle", "sink"]
    assert result["proposals"][1]["task_family"] == "franka_sink_inspection"
    assert result["proposals"][1]["binding_view_id"] == "third_person"


def test_local_analyzer_honors_task_context_label_filter(tmp_path, monkeypatch) -> None:
    payload, weights = _payload(tmp_path)
    payload["analyzer_request"]["task_context"] = {"allowed_object_labels": ["sink"]}
    monkeypatch.setattr(
        analyzer, "MODEL_WEIGHT_SHA256", hashlib.sha256(b"pinned-weights").hexdigest()
    )

    result = analyzer.analyze_payload(
        payload,
        weights_path=weights,
        detector=lambda _paths: [
            [
                {"label": "cup", "score": 0.97, "bbox_xyxy_pixels": [1, 2, 3, 4]},
                {"label": "sink", "score": 0.87, "bbox_xyxy_pixels": [5, 6, 7, 8]},
            ]
        ],
    )

    assert [row["object_label"] for row in result["proposals"]] == ["sink"]
    assert result["proposals"][0]["visual_confidence"] == 0.87


def test_local_analyzer_abstains_when_weights_are_missing(tmp_path) -> None:
    payload, _weights = _payload(tmp_path)
    result = analyzer.analyze_payload(payload, weights_path=tmp_path / "missing.pth")
    assert result["status"] == "abstained"
    assert result["blockers"] == ["torchvision_analyzer_weights_missing"]


def test_local_analyzer_abstains_on_weight_digest_mismatch(tmp_path) -> None:
    payload, weights = _payload(tmp_path)
    result = analyzer.analyze_payload(payload, weights_path=weights)
    assert result["status"] == "abstained"
    assert result["blockers"] == ["torchvision_analyzer_weights_digest_mismatch"]


def test_local_analyzer_abstains_on_rgb_replay(tmp_path, monkeypatch) -> None:
    payload, weights = _payload(tmp_path)
    payload["analyzer_request"]["rendered_views"][0]["rgb_digest"] = "sha256:" + "b" * 64
    monkeypatch.setattr(
        analyzer, "MODEL_WEIGHT_SHA256", hashlib.sha256(b"pinned-weights").hexdigest()
    )
    result = analyzer.analyze_payload(payload, weights_path=weights, detector=lambda _paths: [])
    assert result["status"] == "abstained"
    assert result["blockers"] == ["torchvision_analyzer_rgb_digest_mismatch"]


def test_local_analyzer_uses_g1_catalog_for_humanoid(tmp_path, monkeypatch) -> None:
    payload, weights = _payload(tmp_path, robot_id="g1")
    monkeypatch.setattr(
        analyzer, "MODEL_WEIGHT_SHA256", hashlib.sha256(b"pinned-weights").hexdigest()
    )
    result = analyzer.analyze_payload(
        payload,
        weights_path=weights,
        detector=lambda _paths: [
            [{"label": "chair", "score": 0.91, "bbox_xyxy_pixels": [1, 2, 30, 40]}]
        ],
    )
    assert result["status"] == "completed"
    assert result["proposals"][0]["task_family"] == "g1_obstacle_aware_navigation"
