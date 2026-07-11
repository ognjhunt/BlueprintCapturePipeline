from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline import g1_kitchen_digitalocean_closure as closure


def _identity() -> dict[str, str]:
    digest = "a" * 64
    return {
        "run_id": "run-1",
        "attempt_id": "attempt-1",
        "launch_nonce": "nonce-1",
        "source_commit": "b" * 40,
        "image_digest": digest,
        "bundle_digest": digest,
        "kitchen_asset_digest": digest,
        "active_selection_sha256": digest,
        "task_contract_sha256": digest,
        "provider_allocation_id": "do-1",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_startup_rows_require_attempt_bound_nonce_image_and_all_three_gates(
    tmp_path: Path,
) -> None:
    identity = _identity()
    summary = {
        "status": "passed",
        "launch_session_id": identity["launch_nonce"],
        "image_digest": f"registry/image@sha256:{identity['image_digest']}",
        "gates": {
            gate_id: {"status": "passed"}
            for gate_id in (
                "fast_startup_canary",
                "review_renderer_canary",
                "kitchen_asset_startup_gate",
            )
        },
    }
    path = (
        tmp_path
        / "closed_loop_out"
        / "startup_gates"
        / "supervised_startup_gates.json"
    )
    _write_json(path, summary)

    rows = closure._startup_rows(collected_root=tmp_path, identity=identity)

    assert {row["status"] for row in rows.values()} == {"passed"}
    summary["launch_session_id"] = "stale-nonce"
    _write_json(path, summary)
    stale = closure._startup_rows(collected_root=tmp_path, identity=identity)
    assert {row["status"] for row in stale.values()} == {"blocked"}


def test_collected_media_rows_are_authoritative_and_missing_media_blocks(
    tmp_path: Path, monkeypatch
) -> None:
    identity = _identity()
    missing = closure._collected_media_rows(
        collected_root=tmp_path, identity=identity
    )
    assert missing["robot_pov"]["status"] == "blocked"
    assert missing["semantic_review"]["blockers"] == [
        "full_ordered_episode_media_not_collected"
    ]

    frames = tmp_path / "closed_loop_out" / "scenario-1" / "frames"
    frames.mkdir(parents=True)
    for role in ("overview", "robot_pov"):
        (frames / f"{role}_0000.png").write_bytes(b"frame")
    calls: list[tuple[Path, int]] = []

    def fake_admit(*, scenario_dir, expected_frame_count):
        calls.append((Path(scenario_dir), expected_frame_count))
        return {"status": "passed", "blockers": [], "full_ordered_episode_admitted": True}

    monkeypatch.setattr(closure, "admit_collected_scenario_episode", fake_admit)
    admitted = closure._collected_media_rows(
        collected_root=tmp_path, identity=identity
    )

    assert calls == [(frames.parent, 1)]
    assert admitted["robot_pov"]["status"] == "passed"
    assert admitted["semantic_review"]["status"] == "passed"
