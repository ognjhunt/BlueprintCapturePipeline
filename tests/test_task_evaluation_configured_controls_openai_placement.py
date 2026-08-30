from __future__ import annotations

import fcntl
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from blueprint_pipeline import (
    task_evaluation_configured_controls_openai_placement as placement,
)


NOW = datetime(2026, 8, 30, 14, 0, tzinfo=UTC)


def _file(path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    path.chmod(0o600)
    return path.resolve()


def _environment(tmp_path: Path) -> dict[str, str]:
    visual_key = _file(tmp_path / "secrets" / "visual-key", "test-key\n")
    admin_key = _file(tmp_path / "secrets" / "admin-key", "test-admin\n")
    attestation = _file(tmp_path / "secrets" / "scope.json", "{}\n")
    return {
        "OPENAI_PROJECT_ID": "proj_scene839873",
        "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID": "key_visual_review",
        "OPENAI_API_KEY_FILE": str(visual_key),
        "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE": str(visual_key),
        "OPENAI_ADMIN_API_KEY_FILE": str(admin_key),
        "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE": str(
            attestation
        ),
        "VAST_LAUNCH_LOCK_FILE": str(tmp_path / "locks" / "vast.lock"),
        "BLUEPRINT_VAST_MAX_CONCURRENT_PAID_LAUNCHES": "3",
    }


def _authority() -> dict[str, object]:
    return {
        "provider_id": "openai",
        "credential_role": "artifixer_visual_review",
        "project_id": "proj_scene839873",
        "api_key_id": "key_visual_review",
        "paid_resource_class": (
            "task_evaluation_configured_controls_robot_placement"
        ),
        "maximum_cost_usd": 2.56,
    }


def test_existing_visual_review_key_builds_exact_official_cost_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        placement,
        "build_openai_official_cost_run_gate",
        lambda **kwargs: kwargs,
    )

    built = placement.configured_controls_robot_placement_openai_gate(
        environment=_environment(tmp_path),
        placement_authority=_authority(),
        run_id="scene839873-agent-placement",
        request_digest="sha256:" + "1" * 64,
        candidate_digest="sha256:" + "2" * 64,
        authorization_receipt_digest="sha256:" + "3" * 64,
        output_root=tmp_path / "cost",
        wall_clock=lambda: NOW,
    )

    assert built["project_id"] == "proj_scene839873"
    assert built["api_key_id"] == "key_visual_review"
    assert built["paid_resource_class"] == placement.PAID_RESOURCE_CLASS
    assert built["max_cost_usd"] == 2.56
    assert built["require_zero_baseline"] is False
    resolved = json.loads(
        Path(built["scope_attestation_path"]).read_text(encoding="utf-8")
    )
    assert resolved["paid_resource_class"] == placement.PAID_RESOURCE_CLASS
    assert resolved["project_id"] == "proj_scene839873"
    assert resolved["api_key_id"] == "key_visual_review"
    assert resolved["exclusive_use"] is True


def test_generic_or_wrong_key_cannot_satisfy_the_visual_review_role(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    environment["OPENAI_API_KEY_FILE"] = str(
        _file(tmp_path / "secrets" / "generic-key", "different-test-key\n")
    )

    with pytest.raises(
        placement.TaskEvaluationConfiguredControlsOpenAIPlacementError,
        match="configured_controls_openai_credential_role_mismatch",
    ):
        placement.configured_controls_robot_placement_openai_gate(
            environment=environment,
            placement_authority=_authority(),
            run_id="scene839873-agent-placement",
            request_digest="sha256:" + "1" * 64,
            candidate_digest="sha256:" + "2" * 64,
            authorization_receipt_digest="sha256:" + "3" * 64,
            output_root=tmp_path / "cost",
            wall_clock=lambda: NOW,
        )


def test_visual_review_scope_holds_and_releases_every_vast_slot(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    output_root = tmp_path / "lock-evidence"

    with placement.exclusive_visual_review_cost_scope(
        environment=environment,
        output_root=output_root,
    ) as receipt:
        assert receipt["all_vast_launch_slots_held"] is True
        assert len(receipt["vast_launch_slots_held"]) == 3
        for raw_path in receipt["vast_launch_slots_held"]:
            with Path(raw_path).open("a+", encoding="utf-8") as contender:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(
                        contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                    )

    release = json.loads(
        (output_root / "openai_scope_lock_released.v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert release["status"] == "released"
    assert release["all_vast_launch_slots_released"] is True
    for raw_path in release["vast_launch_slots_released"]:
        with Path(raw_path).open("a+", encoding="utf-8") as contender:
            fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(contender.fileno(), fcntl.LOCK_UN)


def test_busy_vast_slot_refuses_openai_call_and_releases_partial_locks(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    paths = placement._lock_paths(environment)
    paths[1].parent.mkdir(parents=True, exist_ok=True)
    with paths[1].open("a+", encoding="utf-8") as occupied:
        fcntl.flock(occupied.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            placement.TaskEvaluationConfiguredControlsOpenAIPlacementError,
            match="configured_controls_openai_visual_review_scope_busy",
        ):
            with placement.exclusive_visual_review_cost_scope(
                environment=environment,
                output_root=tmp_path / "lock-evidence",
            ):
                pytest.fail("a competing paid lane must prevent model spend")

        with paths[0].open("a+", encoding="utf-8") as first_slot:
            fcntl.flock(first_slot.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(first_slot.fileno(), fcntl.LOCK_UN)

