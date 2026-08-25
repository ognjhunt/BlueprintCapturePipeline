from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
import blueprint_pipeline.native_task_arena_paid_authority as paid
import blueprint_pipeline.native_task_arena_policy_campaign as campaign_module
import blueprint_pipeline.native_task_arena_vast as native_vast
import blueprint_pipeline.task_evaluation_launch_dispatcher as dispatcher
import blueprint_pipeline.vast_provider_adapter as vast
from blueprint_pipeline.task_evaluation_immutable_input_resolver import (
    STAGING_RECEIPT_ENV,
    resolve_immutable_input,
)


COMMIT = "a" * 40
PI_RESOURCE = "blueprint-native-task-policy-pi05-" + "a" * 32
GROOT_RESOURCE = "blueprint-native-task-policy-groot-" + "b" * 32


def _record(path: Path) -> dict[str, object]:
    import hashlib

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    prior_before: float = 38.0,
    reconciled_total: float = 0.25,
    pi_cap: float = 0.5,
    groot_cap: float = 0.5,
    pi_rate: float = 0.5,
    groot_rate: float = 0.5,
    pi_ttl: int = 3600,
    groot_ttl: int = 3600,
    project_total: float | None = None,
    groot_projection_changes: dict[str, object] | None = None,
) -> tuple[Path, dict[str, object], dict[str, dict[str, object]]]:
    bundle_rows: dict[str, dict[str, object]] = {}
    for candidate in campaign_module.MEMBER_IDS:
        root = tmp_path / candidate
        root.mkdir(parents=True)
        bundle = root / "native_task_arena_provider_bundle.zip"
        scene = {
            "schema_version": "native_task_arena_scene_plan.v1",
            "scene_id": "840920",
            "task_id": "840920_task_a_washer_door_open",
            "task_spec": {"prompt": "open the washer door", "maximum_action_steps": 300},
            "scenario": {"cell_id": "canonical-diagnostic-cell"},
            "cadence": {
                "control_frequency_hz": 15.0,
                "physics_frequency_hz": 60.0,
                "maximum_action_steps": 300,
                "episode_length_seconds": 20.0,
            },
        }
        spec = {
            "candidate_id": candidate,
            "task_id": "840920_task_a_washer_door_open",
            "cell_id": "canonical-diagnostic-cell",
            "prompt": "open the washer door",
            "construction_result_digest": "sha256:" + "8" * 64,
            "control_result_digest": "sha256:" + "9" * 64,
            "control_pair_digest": "sha256:" + "a" * 64,
            "max_policy_queries": 20,
            "open_loop_horizon": 15,
            "execution_authority": "development_diagnostic_only",
            "claim_ceiling": "policy_diagnostic_only",
            "initial_state": "canonical_scene_reset",
            "controls_qualified": False,
            "zero_action_negative_bound_separately": True,
        }
        if candidate == "groot_n17_droid":
            for key, value in (groot_projection_changes or {}).items():
                if key.startswith("scene."):
                    scene[key.split(".", 1)[1]] = value
                elif key.startswith("cadence."):
                    scene["cadence"][key.split(".", 1)[1]] = value
                elif key.startswith("scenario."):
                    scene["scenario"][key.split(".", 1)[1]] = value
                else:
                    spec[key] = value
        with zipfile.ZipFile(bundle, "w") as archive:
            archive.writestr(campaign_module._SCENE_PLAN_ARCHIVE_PATH, json.dumps(scene))
            archive.writestr(campaign_module._POLICY_SPEC_ARCHIVE_PATH, json.dumps(spec))
        receipt = root / "native_task_arena_provider_bundle_receipt.v1.json"
        bundle_record = _record(bundle)
        write_json(
            receipt,
            {
                "execution_mode": "policy_diagnostic",
                "policy_candidate_id": candidate,
                "bundle_path": str(bundle),
                "bundle_size_bytes": bundle_record["size_bytes"],
                "bundle_sha256": bundle_record["sha256"],
            },
        )
        bundle_rows[candidate] = {
            "schema_version": "native_task_arena_provider_bundle.v1",
            "status": "ready",
            "execution_mode": "policy_diagnostic",
            "policy_candidate_id": candidate,
            "candidate_policy_queried": False,
            "expected_output_filename": ("native_task_arena_policy_diagnostic_result.v1.json"),
            "implementation_commit": COMMIT,
            "container_image": "nvcr.io/nvidia/isaac-sim@sha256:" + "1" * 64,
            "bundle_path": str(bundle),
            "bundle_sha256": "sha256:" + ("2" if candidate == "pi05_droid" else "3") * 64,
            "input_digest": "sha256:" + ("4" if candidate == "pi05_droid" else "5") * 64,
            "packet_receipt_digest": "sha256:" + "6" * 64,
            "scene_id": "840920",
            "task_id": (
                str(groot_projection_changes.get("receipt_task_id"))
                if candidate == "groot_n17_droid"
                and groot_projection_changes
                and "receipt_task_id" in groot_projection_changes
                else "840920_task_a_washer_door_open"
            ),
            "request_digest": "sha256:" + "1" * 64,
            "arena_scene_plan_digest": "sha256:" + "2" * 64,
            "runtime_contract_digest": "sha256:" + "3" * 64,
            "scenario_instance_digest": "sha256:" + "4" * 64,
            "runtime_source_packet": {
                "receipt_digest": "sha256:" + "7" * 64,
                "packet_sha256": "sha256:" + "b" * 64,
                "packet_size_bytes": 1234,
            },
            "bound_runtime_inputs": [
                {
                    "relative_path": "runtime_inputs/native_task_arena_construction_result.v1.json",
                    "size_bytes": 100,
                    "sha256": "sha256:" + "c" * 64,
                },
                {
                    "relative_path": "runtime_inputs/native_task_arena_control_result.v1.json",
                    "size_bytes": 200,
                    "sha256": "sha256:" + "d" * 64,
                },
            ],
            "receipt_path": receipt,
        }
        if candidate == "groot_n17_droid" and groot_projection_changes:
            row = bundle_rows[candidate]
            for key, value in groot_projection_changes.items():
                if key in {
                    "scene_id",
                    "packet_receipt_digest",
                    "arena_scene_plan_digest",
                    "runtime_contract_digest",
                    "scenario_instance_digest",
                }:
                    row[key] = value
                elif key == "runtime_source_receipt_digest":
                    row["runtime_source_packet"]["receipt_digest"] = value
                elif key == "runtime_source_packet_sha256":
                    row["runtime_source_packet"]["packet_sha256"] = value
                elif key == "construction_record_sha256":
                    row["bound_runtime_inputs"][0]["sha256"] = value
                elif key == "control_record_sha256":
                    row["bound_runtime_inputs"][1]["sha256"] = value

    monkeypatch.setattr(
        campaign_module,
        "_verified_policy_bundle",
        lambda path, **_kwargs: next(
            row for row in bundle_rows.values() if Path(str(row["receipt_path"])) == path
        ),
    )
    prior_root = tmp_path / "prior"
    prior_root.mkdir()
    prior_files: dict[str, Path] = {}
    for name in ("authority", "terminal_result", "provider_zero"):
        path = prior_root / f"{name}.json"
        write_json(path, {"name": name})
        prior_files[name] = path
    prior = {
        "authority_digest": "sha256:" + "8" * 64,
        "aggregate_goal_spend_before_attempt_usd": prior_before,
        "records": {name: _record(path) for name, path in prior_files.items()},
    }
    reconciliation_path = prior_root / "reconciliation.json"
    write_json(reconciliation_path, {"status": "official"})
    reconciled = {
        "prior_terminal_attempts": [{"result": _record(prior_files["terminal_result"])}],
        "reconciliation": _record(reconciliation_path),
        "actual_total_usd": reconciled_total,
    }
    monkeypatch.setattr(campaign_module, "validate_terminal_spend_chain", lambda **_kwargs: prior)
    monkeypatch.setattr(campaign_module, "bind_lane_prior_spend", lambda **_kwargs: reconciled)
    project_path = prior_root / "project-spend.json"
    if project_total is not None:
        write_json(project_path, {"total_cost_usd": project_total})
        monkeypatch.setattr(
            campaign_module,
            "validate_project_spend_reconciliation",
            lambda *_args, **_kwargs: (
                {"total_cost_usd": project_total},
                _record(project_path),
            ),
        )
    output = tmp_path / "policy-campaign.json"
    value = campaign_module.materialize_native_task_arena_policy_campaign(
        campaign_id="scene-840920-policy-diagnostic-pair-1",
        blueprint_commit=COMMIT,
        pi05_bundle_receipt_path=bundle_rows["pi05_droid"]["receipt_path"],
        groot_bundle_receipt_path=bundle_rows["groot_n17_droid"]["receipt_path"],
        prior_authority_path=prior_files["authority"],
        prior_result_path=prior_files["terminal_result"],
        prior_provider_zero_path=prior_files["provider_zero"],
        prior_spend_reconciliation_path=reconciliation_path,
        project_spend_reconciliation_path=(
            project_path if project_total is not None else None
        ),
        controls_allowed_active_instance_ids=[48610674],
        pi05_launch_id="adp-policy-pi05-campaign-1",
        pi05_resource_name=PI_RESOURCE,
        pi05_max_hourly_rate_usd=pi_rate,
        pi05_hard_cap_usd=pi_cap,
        pi05_hard_ttl_seconds=pi_ttl,
        groot_launch_id="adp-policy-groot-campaign-1",
        groot_resource_name=GROOT_RESOURCE,
        groot_max_hourly_rate_usd=groot_rate,
        groot_hard_cap_usd=groot_cap,
        groot_hard_ttl_seconds=groot_ttl,
        output_path=output,
    )
    return output, value, bundle_rows


def test_campaign_uses_newer_conservative_project_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _path, campaign, _bundles = _campaign(
        tmp_path,
        monkeypatch,
        prior_before=38.0,
        reconciled_total=0.25,
        project_total=43.197914,
    )

    assert campaign["prior_official_spend"][
        "aggregate_goal_spend_before_campaign_usd"
    ] == 43.197914
    assert campaign["projected_aggregate_goal_spend_usd"] == 44.197914
    assert campaign["prior_official_spend"][
        "project_spend_reconciliation"
    ]["path"].endswith("project-spend.json")


def test_campaign_rejects_project_total_older_than_terminal_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(
        ValueError, match="native_task_arena_policy_campaign_project_spend_stale"
    ):
        _campaign(
            tmp_path,
            monkeypatch,
            prior_before=38.0,
            reconciled_total=0.25,
            project_total=38.0,
        )


def test_two_member_campaign_binds_both_caps_and_member_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_path, campaign, bundles = _campaign(tmp_path, monkeypatch)

    assert campaign["maximum_campaign_spend_usd"] == 1.0
    assert campaign["projected_aggregate_goal_spend_usd"] == 39.25
    assert [row["candidate_id"] for row in campaign["members"]] == [
        "pi05_droid",
        "groot_n17_droid",
    ]
    prepared = bundles["pi05_droid"]
    binding = paid._native_policy_campaign_binding(
        campaign_path=campaign_path,
        campaign_member_id="pi05_droid",
        prepared_bundle=prepared,
        blueprint_commit=COMMIT,
        prior_spend=38.25,
        max_hourly_rate_usd=0.5,
        hard_cap_usd=0.5,
        hard_ttl_seconds=3600,
        allowed_active_instance_ids=[48610674],
    )

    assert binding["campaign_digest"] == campaign["campaign_digest"]
    assert binding["member_id"] == "pi05_droid"
    assert binding["launch_id"] == "adp-policy-pi05-campaign-1"
    assert binding["sibling_resource_name"] == GROOT_RESOURCE


def test_campaign_accepts_rate_above_cap_when_ttl_projection_fits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_path, campaign, _bundles = _campaign(
        tmp_path,
        monkeypatch,
        pi_rate=0.64,
        groot_rate=0.64,
        pi_cap=0.5,
        groot_cap=0.5,
        pi_ttl=2_800,
        groot_ttl=2_800,
    )

    assert campaign_path.is_file()
    assert campaign_module.validate_native_task_arena_policy_campaign(campaign)[
        "maximum_campaign_spend_usd"
    ] == 1.0


def test_campaign_rejects_ttl_projection_above_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(
        ValueError, match="native_task_arena_policy_campaign_member_invalid"
    ):
        _campaign(
            tmp_path,
            monkeypatch,
            pi_rate=0.64,
            groot_rate=0.64,
            pi_cap=0.5,
            groot_cap=0.5,
            pi_ttl=2_813,
            groot_ttl=2_813,
        )


def test_campaign_validator_reads_only_staged_receipts_and_bundles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_path, campaign, bundles = _campaign(tmp_path, monkeypatch)
    declared = [campaign_path]
    for member in campaign["members"]:
        receipt_path = Path(member["bundle_receipt"]["path"])
        declared.extend(
            [
                receipt_path,
                Path(json.loads(receipt_path.read_text())["bundle_path"]),
            ]
        )
    profile = {
        "profile_id": "campaign-staging-test",
        "profile_digest": "sha256:" + "0" * 64,
        "immutable_inputs": [
            {
                "name": f"input-{index}",
                "path": str(path.resolve()),
                "digest": _record(path)["sha256"],
            }
            for index, path in enumerate(declared)
        ],
    }
    staging, rewritten = dispatcher._stage_profile_immutable_inputs(
        profile=profile,
        run_root=tmp_path / "run",
        allocator_argv=[str(campaign_path)],
    )
    monkeypatch.setenv(
        STAGING_RECEIPT_ENV,
        str(tmp_path / "run" / "immutable_input_staging_receipt.json"),
    )
    original_paths = set(declared)

    def verified_staged_bundle(path: Path, **_kwargs: object) -> dict[str, object]:
        assert path not in original_paths
        receipt = json.loads(path.read_text())
        bundle = resolve_immutable_input(
            receipt["bundle_path"],
            expected_digest=receipt["bundle_sha256"],
            expected_size_bytes=receipt["bundle_size_bytes"],
        )
        assert bundle not in original_paths
        candidate = receipt["policy_candidate_id"]
        return bundles[candidate]

    monkeypatch.setattr(
        campaign_module, "_verified_policy_bundle", verified_staged_bundle
    )
    for path in declared:
        path.write_bytes(b"tampered-after-staging")

    staged_campaign = Path(rewritten[0])
    binding = paid._native_policy_campaign_binding(
        campaign_path=staged_campaign,
        campaign_member_id="pi05_droid",
        prepared_bundle=bundles["pi05_droid"],
        blueprint_commit=COMMIT,
        prior_spend=38.25,
        max_hourly_rate_usd=0.5,
        hard_cap_usd=0.5,
        hard_ttl_seconds=3600,
        allowed_active_instance_ids=[48610674],
    )

    assert binding["campaign_digest"] == campaign["campaign_digest"]
    assert staging["input_count"] == len(declared)


def test_single_use_authority_binds_exact_campaign_member_and_rejects_alteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_path, campaign, bundles = _campaign(tmp_path, monkeypatch)
    prepared = bundles["pi05_droid"]
    prior_root = tmp_path / "authority-prior"
    prior_root.mkdir()
    prior_paths: dict[str, Path] = {}
    for name in ("authority", "terminal_result", "provider_zero"):
        path = prior_root / f"{name}.json"
        write_json(path, {"name": name})
        prior_paths[name] = path
    reconciliation_path = prior_root / "reconciliation.json"
    write_json(reconciliation_path, {"status": "official"})
    prior = {
        "authority_digest": "sha256:" + "8" * 64,
        "attempt_cost_usd": 0.1,
        "aggregate_goal_spend_before_attempt_usd": 38.0,
        "records": {name: _record(path) for name, path in prior_paths.items()},
    }
    reconciled = {
        "prior_terminal_attempts": [{"result": _record(prior_paths["terminal_result"])}],
        "reconciliation": _record(reconciliation_path),
        "actual_total_usd": 0.25,
    }
    monkeypatch.setattr(
        paid,
        "_bundle_loader",
        lambda _mode: lambda *_args, **_kwargs: prepared,
    )
    monkeypatch.setattr(paid, "validate_terminal_spend_chain", lambda **_kwargs: prior)
    monkeypatch.setattr(paid, "bind_lane_prior_spend", lambda **_kwargs: reconciled)
    monkeypatch.setattr(
        paid, "validate_bound_lane_prior_spend", lambda *_args, **_kwargs: reconciled
    )
    receipt = Path(str(prepared["receipt_path"]))
    authority = paid.materialize_native_task_arena_paid_attempt_authority(
        bundle_receipt_path=receipt,
        prior_authority_path=prior_paths["authority"],
        prior_result_path=prior_paths["terminal_result"],
        prior_provider_zero_path=prior_paths["provider_zero"],
        prior_spend_reconciliation_path=reconciliation_path,
        authorization_reference="user-directed paired policy diagnostics",
        authorized_by="user",
        authorized_on="2026-08-25",
        blueprint_commit=COMMIT,
        max_hourly_rate_usd=0.5,
        hard_cap_usd=0.5,
        hard_ttl_seconds=3600,
        output_path=tmp_path / "pi05-authority.json",
        allowed_active_instance_ids=[48610674],
        policy_campaign_path=campaign_path,
        campaign_member_id="pi05_droid",
    )

    binding = authority["policy_campaign_binding"]
    assert binding["campaign_digest"] == campaign["campaign_digest"]
    assert binding["member_id"] == "pi05_droid"
    assert authority["maximum_automatic_retries"] == 0
    assert authority["maximum_provider_allocations"] == 1
    altered = json.loads(json.dumps(authority))
    altered["policy_campaign_binding"]["member_id"] = "groot_n17_droid"
    altered["authorization_digest"] = canonical_digest(altered, digest_field="authorization_digest")
    with pytest.raises(ValueError, match="native_task_arena_authority_invalid"):
        paid.validate_native_task_arena_paid_attempt_authority(
            altered,
            prepared_bundle=prepared,
            max_hourly_rate_usd=0.5,
            hard_cap_usd=0.5,
            hard_ttl_seconds=3600,
            allowed_active_instance_ids=[48610674],
        )


def test_campaign_rejects_sum_of_member_caps_over_fifty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(
        ValueError, match="native_task_arena_policy_campaign_aggregate_spend_invalid"
    ):
        _campaign(
            tmp_path,
            monkeypatch,
            prior_before=49.0,
            reconciled_total=0.25,
        )
    assert not (tmp_path / "policy-campaign.json").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"scene_id": "840921"},
        {
            "receipt_task_id": "other-task",
            "scene.task_id": "other-task",
            "task_id": "other-task",
        },
        {"scenario.cell_id": "other-cell", "cell_id": "other-cell"},
        {"arena_scene_plan_digest": "sha256:" + "e" * 64},
        {"runtime_contract_digest": "sha256:" + "e" * 64},
        {"scenario_instance_digest": "sha256:" + "e" * 64},
        {"runtime_source_receipt_digest": "sha256:" + "e" * 64},
        {"runtime_source_packet_sha256": "sha256:" + "e" * 64},
        {
            "construction_record_sha256": "sha256:" + "e" * 64,
            "construction_result_digest": "sha256:" + "e" * 64,
        },
        {
            "control_record_sha256": "sha256:" + "e" * 64,
            "control_result_digest": "sha256:" + "e" * 64,
        },
        {"control_pair_digest": "sha256:" + "e" * 64},
        {"claim_ceiling": "different-diagnostic-predecessor"},
        {"max_policy_queries": 19},
        {"open_loop_horizon": 14},
        {"cadence.maximum_action_steps": 299},
    ],
)
def test_campaign_rejects_mismatched_shared_science(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, object],
) -> None:
    with pytest.raises(
        ValueError, match="native_task_arena_policy_campaign_shared_science_mismatch"
    ):
        _campaign(tmp_path, monkeypatch, groot_projection_changes=changes)


@pytest.mark.parametrize(
    "bounds",
    [
        {"groot_rate": 0.49},
        {"groot_cap": 0.49, "groot_rate": 0.49},
        {"groot_ttl": 3540},
    ],
)
def test_campaign_rejects_asymmetric_member_resource_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bounds: dict[str, object],
) -> None:
    with pytest.raises(
        ValueError, match="native_task_arena_policy_campaign_member_limits_asymmetric"
    ):
        _campaign(tmp_path, monkeypatch, **bounds)


def test_altered_campaign_and_member_bindings_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_path, campaign, bundles = _campaign(tmp_path, monkeypatch)
    altered = json.loads(json.dumps(campaign))
    altered["members"][0]["bundle_sha256"] = "sha256:" + "f" * 64
    altered["campaign_digest"] = canonical_digest(altered, digest_field="campaign_digest")
    write_json(campaign_path, altered)

    with pytest.raises(
        ValueError, match="native_task_arena_policy_campaign_member_binding_invalid"
    ):
        paid._native_policy_campaign_binding(
            campaign_path=campaign_path,
            campaign_member_id="pi05_droid",
            prepared_bundle=bundles["pi05_droid"],
            blueprint_commit=COMMIT,
            prior_spend=38.25,
            max_hourly_rate_usd=0.5,
            hard_cap_usd=0.5,
            hard_ttl_seconds=3600,
            allowed_active_instance_ids=[48610674],
        )
    with pytest.raises(ValueError, match="native_task_arena_policy_campaign_member_missing"):
        paid._native_policy_campaign_binding(
            campaign_path=campaign_path,
            campaign_member_id="not-a-member",
            prepared_bundle=bundles["pi05_droid"],
            blueprint_commit=COMMIT,
            prior_spend=38.25,
            max_hourly_rate_usd=0.5,
            hard_cap_usd=0.5,
            hard_ttl_seconds=3600,
            allowed_active_instance_ids=[48610674],
        )


def test_campaign_rejects_resource_name_without_strong_member_entropy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _path, campaign, _bundles = _campaign(tmp_path, monkeypatch)
    altered = json.loads(json.dumps(campaign))
    altered["members"][0]["resource_name"] = "blueprint-native-task-policy-pi05-predictable"
    altered["campaign_digest"] = canonical_digest(altered, digest_field="campaign_digest")
    with pytest.raises(ValueError, match="native_task_arena_policy_campaign_member_invalid"):
        campaign_module.validate_native_task_arena_policy_campaign(altered)


def test_campaign_rejects_strong_entropy_outside_watchdog_policy_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _path, campaign, _bundles = _campaign(tmp_path, monkeypatch)
    altered = json.loads(json.dumps(campaign))
    altered["members"][0]["resource_name"] = "blueprint-policy-pi05-" + "c" * 32
    altered["campaign_digest"] = canonical_digest(
        altered, digest_field="campaign_digest"
    )
    with pytest.raises(
        ValueError, match="native_task_arena_policy_campaign_member_invalid"
    ):
        campaign_module.validate_native_task_arena_policy_campaign(altered)


def _inventory(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]) -> None:
    monkeypatch.setattr(
        vast,
        "_api_json",
        lambda **_kwargs: (200, {"instances": rows}),
    )


def test_campaign_inventory_accepts_named_sibling_and_controls_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _inventory(
        monkeypatch,
        [
            {"id": 48610674, "actual_status": "running", "label": "controls"},
            {
                "id": 48620000,
                "actual_status": "running",
                "label": GROOT_RESOURCE,
            },
        ],
    )

    result = vast._prelaunch_inventory_guard(
        job_dir=tmp_path,
        generated_at="fixed",
        api_key="not-recorded",
        allowed_active_instance_ids=[48610674],
        allowed_active_resource_names=[GROOT_RESOURCE],
        lane_label_prefix="blueprint-native-task-policy-diagnostic-",
    )

    assert result["status"] == "passed"
    assert result["allowed_named_active_instance_count"] == 1
    assert result["unexpected_active_instances"] == []


def test_campaign_inventory_rejects_impostor_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _inventory(
        monkeypatch,
        [
            {
                "id": 48620001,
                "actual_status": "running",
                "label": "blueprint-policy-groot-impostor-20260825",
            }
        ],
    )

    result = vast._prelaunch_inventory_guard(
        job_dir=tmp_path,
        generated_at="fixed",
        api_key="not-recorded",
        allowed_active_resource_names=[GROOT_RESOURCE],
        lane_label_prefix="blueprint-native-task-policy-diagnostic-",
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["active_vast_instances_detected_before_new_launch"]
    assert result["unexpected_active_instances"][0]["id"] == 48620001


@pytest.mark.parametrize("label", [GROOT_RESOURCE + "-near-match", GROOT_RESOURCE[:-1] + "c"])
def test_campaign_inventory_rejects_same_prefix_near_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str
) -> None:
    _inventory(
        monkeypatch,
        [{"id": 48620002, "actual_status": "running", "label": label}],
    )
    result = vast._prelaunch_inventory_guard(
        job_dir=tmp_path,
        generated_at="fixed",
        api_key="not-recorded",
        allowed_active_resource_names=[GROOT_RESOURCE],
        lane_label_prefix="blueprint-native-task-policy-diagnostic-",
    )
    assert result["status"] == "blocked"
    assert result["unexpected_active_instance_count"] == 1


def test_default_non_campaign_inventory_behavior_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _inventory(
        monkeypatch,
        [
            {
                "id": 90,
                "actual_status": "running",
                "label": "blueprint-unrelated-lane-1",
            }
        ],
    )
    result = vast._prelaunch_inventory_guard(
        job_dir=tmp_path,
        generated_at="fixed",
        api_key="not-recorded",
        lane_label_prefix="blueprint-native-task-policy-diagnostic-",
    )

    assert result["status"] == "passed"
    assert result["campaign_name_scope_active"] is False
    assert result["foreign_active_instance_count"] == 1


def test_policy_transport_forwards_campaign_names_without_replacing_safety_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = {
        "schema_version": "native_task_arena_provider_bundle.v1",
        "execution_mode": "policy_diagnostic",
        "policy_candidate_id": "pi05_droid",
        "candidate_policy_queried": False,
        "expected_output_filename": ("native_task_arena_policy_diagnostic_result.v1.json"),
        "container_image": "nvcr.io/nvidia/isaac-sim@sha256:" + "1" * 64,
    }
    authority = {
        "policy_campaign_binding": {
            "resource_name": PI_RESOURCE,
            "sibling_resource_name": GROOT_RESOURCE,
        }
    }
    monkeypatch.setattr(
        native_vast,
        "validate_native_task_arena_paid_attempt_authority",
        lambda *_args, **_kwargs: authority,
    )
    observed: dict[str, object] = {}

    def run_transport(**kwargs):  # type: ignore[no-untyped-def]
        observed.update(kwargs)
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(native_vast, "run_arena_native_control_vast", run_transport)

    result = native_vast.run_native_task_arena_policy_diagnostic_vast(
        job_dir=tmp_path,
        prepared_bundle=prepared,
        paid_resource_admission_grant=None,
        execute=False,
        allowed_active_instance_ids=[48610674],
        paid_attempt_authority=authority,
    )

    assert result["status"] == "dry_run_ready"
    assert observed["instance_label_prefix"] == "blueprint-native-task-policy-diagnostic-"
    assert observed["instance_label_exact"] == PI_RESOURCE
    assert observed["allowed_active_resource_names"] == (GROOT_RESOURCE,)
    assert observed["stale_offer_create_retry_limit"] == 0
    assert observed["vast_launch_lock_file"] is None
    assert observed["require_independent_watchdog"] is True


def test_non_campaign_policy_transport_also_forces_zero_stale_offer_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = {
        "schema_version": "native_task_arena_provider_bundle.v1",
        "execution_mode": "policy_diagnostic",
        "policy_candidate_id": "pi05_droid",
        "candidate_policy_queried": False,
        "expected_output_filename": "native_task_arena_policy_diagnostic_result.v1.json",
        "container_image": "nvcr.io/nvidia/isaac-sim@sha256:" + "1" * 64,
    }
    authority: dict[str, object] = {}
    monkeypatch.setattr(
        native_vast,
        "validate_native_task_arena_paid_attempt_authority",
        lambda *_args, **_kwargs: authority,
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        native_vast,
        "run_arena_native_control_vast",
        lambda **kwargs: observed.update(kwargs) or {"status": "dry_run_ready"},
    )
    native_vast.run_native_task_arena_policy_diagnostic_vast(
        job_dir=tmp_path,
        prepared_bundle=prepared,
        paid_resource_admission_grant=None,
        execute=False,
        paid_attempt_authority=authority,
    )
    assert observed["instance_label_exact"] is None
    assert observed["allowed_active_resource_names"] == ()
    assert observed["stale_offer_create_retry_limit"] == 0
