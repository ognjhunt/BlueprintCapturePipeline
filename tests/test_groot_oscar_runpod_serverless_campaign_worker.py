import hashlib
import json
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import groot_oscar_runpod_serverless_campaign_worker as campaign


SOURCE = "c" * 40
IMAGE = "docker.io/example/worker@sha256:" + "d" * 64
MODEL = "sha256:" + "e" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _campaign_input(root: Path) -> tuple[Path, str]:
    bundle = root / "inputs" / "payload.zip"
    bundle.parent.mkdir(parents=True)
    with zipfile.ZipFile(bundle, "w") as archive:
        for name in (
            "initial_policy_frame.png",
            "route.json",
            "task_prompt.txt",
            "task_success_contract.json",
            "kitchen_asset_inventory_checksums.json",
            "kitchen/KitchenRoom.usd",
        ):
            archive.writestr(name, "move the simulated object" if name.endswith(".txt") else "{}")
    bundle_sha = _sha256(bundle)
    empty_json_sha = hashlib.sha256(b"{}").hexdigest()
    attempts = []
    for attempt_id, kind, seed, timeout in campaign.EXPECTED_ATTEMPTS:
        manifest = root / "inputs" / f"{attempt_id}.json"
        _write_json(
            manifest,
            {
                "schema_version": campaign.ATTEMPT_INPUT_SCHEMA_VERSION,
                "attempt_id": attempt_id,
                "source_commit": SOURCE,
                "source_dirty_patch_sha256": (
                    campaign.CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256
                ),
                "image_digest": IMAGE.rsplit("@", 1)[-1],
                "artifacts": {
                    "bundle": {"sha256": bundle_sha},
                    "route": {"sha256": empty_json_sha},
                    "task_success_contract": {"sha256": empty_json_sha},
                    "kitchen_inventory": {"sha256": empty_json_sha},
                    "selection": {"sha256": "a" * 64},
                },
                "serverless_runtime_qualification_contract": {
                    "schema_version": (
                        "g1_kitchen_serverless_runtime_qualification.v1"
                    ),
                    "startup_reverified_in_campaign_job": True,
                    "strict_three_action_probe_required_before_campaign": True,
                    "same_runtime_worker_identity_required": True,
                },
            },
        )
        attempts.append(
            {
                "attempt_id": attempt_id,
                "kind": kind,
                "seed": seed,
                "timeout_seconds": timeout,
                "attempt_manifest": {
                    "relative_path": manifest.relative_to(root).as_posix(),
                    "sha256": _sha256(manifest),
                },
            }
        )
    manifest = root / "inputs" / "campaign.json"
    _write_json(
        manifest,
        {
            "schema_version": campaign.INPUT_SCHEMA_VERSION,
            "campaign_id": "campaign-test",
            "source_commit": SOURCE,
            "worker_image_ref": IMAGE,
            "model_manifest_digest": MODEL,
            "runtime": {
                "dynamic_episode_termination": True,
                "stop_immediately_on_declared_completion": True,
                "fixed_frame_count": None,
                "review_width": 640,
                "review_height": 480,
            },
            "payload_bundle": {
                "relative_path": bundle.relative_to(root).as_posix(),
                "sha256": bundle_sha,
            },
            "attempts": attempts,
        },
    )
    return manifest, _sha256(manifest)


def test_campaign_input_requires_exact_dynamic_attempt_contract() -> None:
    payload = {
        "schema_version": campaign.INPUT_SCHEMA_VERSION,
        "source_commit": SOURCE,
        "worker_image_ref": IMAGE,
        "model_manifest_digest": MODEL,
        "runtime": {
            "dynamic_episode_termination": True,
            "stop_immediately_on_declared_completion": True,
            "fixed_frame_count": 48,
            "review_width": 640,
            "review_height": 480,
        },
        "attempts": [],
    }

    with pytest.raises(ValueError, match="campaign_runtime_contract_invalid"):
        campaign._validate_campaign_input(
            payload,
            source_commit=SOURCE,
            image_ref=IMAGE,
            model_manifest_digest=MODEL,
        )


def test_extract_zip_rejects_symlink_member(tmp_path: Path) -> None:
    source = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(source, "w") as archive:
        member = zipfile.ZipInfo("escape")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(member, "../outside")

    with pytest.raises(ValueError, match="campaign_bundle_unsafe_member"):
        campaign._extract_zip(source, tmp_path / "out")


def test_campaign_runs_smoke_then_three_seeds_and_keeps_semantics_separate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, digest = _campaign_input(tmp_path)
    observed_plan = {}

    def fake_plan(**kwargs):
        observed_plan.update(kwargs)
        return {
            "sealed_active": True,
            "blockers": [],
            "env": {},
            "policy_server_port": 5550,
            "groot_server_command": ["groot"],
            "gear_sonic_controller_command": ["gear"],
        }

    class FakeProcess:
        def poll(self):
            return None

    def fake_start(_command, *, log_path, env):
        del env
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return FakeProcess(), log_path.open("ab")

    run_order = []

    def fake_attempt(*, attempt, artifact_root, **_kwargs):
        run_order.append((attempt["attempt_id"], attempt["seed"], attempt["timeout_seconds"]))
        episode = artifact_root / attempt["attempt_id"]
        episode.mkdir(parents=True)
        (episode / "proof.txt").write_text("artifact", encoding="utf-8")
        return {
            "schema_version": campaign.ATTEMPT_SCHEMA_VERSION,
            "attempt_id": attempt["attempt_id"],
            "kind": attempt["kind"],
            "seed": attempt["seed"],
            "timeout_seconds": attempt["timeout_seconds"],
            "status": "completed",
            "blockers": [],
            "semantic_task_success": False,
        }

    monkeypatch.setattr(campaign, "build_sealed_launch_plan", fake_plan)
    monkeypatch.setattr(campaign, "WORKSPACE_ROOT", tmp_path / "workspace")
    monkeypatch.setattr(campaign, "_start", fake_start)
    monkeypatch.setattr(campaign, "_wait_tcp", lambda *_args: True)
    monkeypatch.setattr(campaign, "_wait_gear", lambda *_args: True)
    monkeypatch.setattr(campaign, "_terminate", lambda *_args: None)
    monkeypatch.setattr(campaign, "_run_attempt", fake_attempt)
    monkeypatch.setattr(
        campaign.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setenv("BLUEPRINT_PROVIDER_ALLOCATION_ID", "worker-test")

    result = campaign.run_kitchen_campaign(
        network_volume_root=tmp_path,
        campaign_manifest_relative_path=manifest.relative_to(tmp_path).as_posix(),
        campaign_manifest_sha256=digest,
        output_relative_path="outputs/campaign-test",
        source_commit=SOURCE,
        image_ref=IMAGE,
        model_manifest_digest=MODEL,
    )

    assert observed_plan["require_forward_inverse_consistency"] is False
    assert observed_plan["allow_wam_consistency_scoring"] is False
    assert run_order == [
        ("smoke", 1000, 300),
        ("episode_001", 1001, 900),
        ("episode_002", 1002, 900),
        ("episode_003", 1003, 900),
    ]
    assert result["status"] == "completed"
    assert result["all_dynamic_episodes_completed"] is True
    assert result["semantic_task_success_by_attempt"] == {
        "smoke": False,
        "episode_001": False,
        "episode_002": False,
        "episode_003": False,
    }
    artifact_root = tmp_path / "outputs" / "campaign-test"
    recorded = {
        row["relative_path"]: row["sha256"]
        for row in result["artifact_manifest"]["files"]
    }
    assert recorded["campaign_result.json"] == _sha256(
        artifact_root / "campaign_result.json"
    )
