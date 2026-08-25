"""The Arena chain has to be orderable from the website, not just runnable.

Three probe kinds over one transport, and they are ordered: controls consumes
the construction result, policy consumes both. The allocator enforces that
itself, but only after a provider has been handed over -- so a profile that
omits a predecessor costs a paid allocation to discover. Refusing at build time
costs nothing.

All three had bundles, a transport, and an allocator branch, and no launch
profile.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.adp009d_policy_rights import build_candidate_policy_rights
from blueprint_pipeline.adp009d_scene_policy_readiness import (
    load_scene_policy_readiness,
)
import blueprint_pipeline.native_task_arena_paid_authority as paid
import blueprint_pipeline.native_task_arena_warm_authority as warm_authority
from blueprint_pipeline.native_task_arena_bundle import (
    POLICY_RUNTIME_ROOT_MODULE_NAMES,
)
from blueprint_pipeline.native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
from blueprint_pipeline.native_task_arena_policy_bundle import (
    ADP009D_POLICY_READINESS_PATH,
    ADP009D_SCENARIO_SUITE_PATH,
    _candidate_runtime_binding,
)
import blueprint_pipeline.task_evaluation_live_profile as live_profile
from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError

pytestmark = pytest.mark.usefixtures(
    "_materialize_generated_manifest_publication_fixture"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
URI = f"https://raw.githubusercontent.com/example/repo/{COMMIT}/arena.json"
SCENE_ID = "840920"
TASK_ID = "task_a_washer_door_open"


def _load():
    name = "build_native_task_arena_live_profile"
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = _load()

_REHEARSAL_SPEC = importlib.util.spec_from_file_location(
    "rehearse_native_task_arena_terminal",
    REPO_ROOT / "scripts/rehearse_lane_terminal_contract.py",
)


def test_native_profile_output_is_exclusive_and_exactly_idempotent(
    tmp_path: Path,
) -> None:
    output = tmp_path / "private-profile.json"
    payload = b'{"profile":"exact"}\n'

    assert builder._write_profile_output_exclusive(output, payload) is True
    assert builder._write_profile_output_exclusive(output, payload) is False
    with pytest.raises(
        TaskEvaluationLaunchError,
        match="native_task_arena_live_profile_output_conflict",
    ):
        builder._write_profile_output_exclusive(
            output, b'{"profile":"different"}\n'
        )
    assert output.read_bytes() == payload
rehearsal = importlib.util.module_from_spec(_REHEARSAL_SPEC)
assert _REHEARSAL_SPEC.loader is not None
_REHEARSAL_SPEC.loader.exec_module(rehearsal)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def _predecessor(root: Path) -> dict[str, Path]:
    root.mkdir()
    authority = {
        "schema_version": "paired_target_native_import_paid_attempt_authority.v1",
        "bundle_sha256": "sha256:" + "b" * 64,
        "hard_attempt_spend_cap_usd": 0.75,
        "maximum_single_resource_ttl_seconds": 3_600,
        "aggregate_goal_spend_before_attempt_usd": 0.0,
        "aggregate_goal_spend_cap_usd": 12.0,
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    authority_path = root / "authority.json"
    write_json(authority_path, authority)
    result = {
        "schema_version": "paired_target_native_import_vast_run.v1",
        "status": "completed",
        "bundle_sha256": authority["bundle_sha256"],
        "estimated_cost_usd": 0.05,
        "hard_cap_usd": 0.75,
        "hard_ttl_seconds": 3_600,
        "retry_cap": 0,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "authorization_consumption": {
            "authorization_digest": authority["authorization_digest"]
        },
    }
    result_path = root / "result.json"
    write_json(result_path, result)
    zero = {
        "schema_version": "paired_target_native_import_provider_zero.v1",
        "status": "completed",
        "attempt_authority_digest": authority["authorization_digest"],
        "terminal_result": _record(result_path),
        "provider_zero_confirmed": True,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "receipt_digest": "",
    }
    zero["receipt_digest"] = canonical_digest(zero, digest_field="receipt_digest")
    zero_path = root / "provider_zero.json"
    write_json(zero_path, zero)
    return {"authority": authority_path, "result": result_path, "zero": zero_path}


def _provider_bundle(
    root: Path,
    *,
    link: str,
    packet_digest: str,
    runtime_source_digest: str,
    request_digest: str,
    scene_plan_digest: str,
    bound_paths: dict[str, Path],
) -> Path:
    root.mkdir()
    bundle = root / "native_task_arena_provider_bundle.zip"
    mode = {
        "construction": "construction_canary",
        "controls": "controls",
        "policy": "policy",
        "policy-diagnostic": "policy_diagnostic",
    }[link]
    if link in {"policy", "policy-diagnostic"}:
        with zipfile.ZipFile(bundle, "w") as archive:
            archive.writestr(
                "provider_runtime/runtime_inputs/"
                "native_task_arena_policy_execution_spec.v1.json",
                bound_paths[
                    "native_task_arena_policy_execution_spec.v1.json"
                ].read_bytes(),
            )
    else:
        bundle.write_bytes(f"production-shaped-{link}-bundle".encode())
    bound_names = {
        "construction": (),
        "controls": (
            "native_task_arena_construction_result.v1.json",
            "adp_task_control_plan.v1.json",
            "adp_task_control_execution_spec.v1.json",
        ),
        "policy": (
            "adp009d_scene_840920_policy_readiness.v1.json",
            "third_scene_840920_task_a_scenario_suite.v1.json",
            "native_task_arena_construction_result.v1.json",
            "native_task_arena_control_result.v1.json",
            "native_task_arena_policy_execution_spec.v1.json",
            "openpi_polaris_checkpoint_inventory.json",
        ),
        "policy-diagnostic": (
            "adp009d_scene_840920_policy_readiness.v1.json",
            "third_scene_840920_task_a_scenario_suite.v1.json",
            "native_task_arena_construction_result.v1.json",
            "native_task_arena_control_result.v1.json",
            "native_task_arena_policy_execution_spec.v1.json",
            "openpi_polaris_checkpoint_inventory.json",
        ),
    }[link]
    manifest = {
        "schema_version": "native_task_arena_provider_bundle.v1",
        "generated_at": "2026-08-16T00:00:00+00:00",
        "status": "ready",
        "program_id": "arm-decision-proof-v1",
        "execution_mode": mode,
        "implementation_commit": COMMIT,
        "container_image": NATIVE_TASK_ARENA_IMAGE,
        "scene_id": SCENE_ID,
        "task_id": TASK_ID,
        "request_digest": request_digest,
        "packet_receipt_digest": packet_digest,
        "arena_scene_plan_digest": scene_plan_digest,
        "runtime_contract_digest": "sha256:" + "2" * 64,
        "scenario_instance_digest": "sha256:" + "3" * 64,
        "packet_files": [],
        "packet_file_count": 0,
        "worker_source_sha256": "sha256:" + "4" * 64,
        "runtime_modules": [],
        "runtime_root_modules": (
            [
                {
                    "relative_path": name,
                    "size_bytes": 1,
                    "sha256": "sha256:" + "8" * 64,
                }
                for name in POLICY_RUNTIME_ROOT_MODULE_NAMES
            ]
            if link in {"policy", "policy-diagnostic"}
            else []
        ),
        "policy_provisioning_script": (
            "adp009d_policy_provisioning.pi05_droid.sh"
            if link in {"policy", "policy-diagnostic"}
            else None
        ),
        "policy_provisioning": (
            {
                "relative_path": "adp009d_policy_provisioning.pi05_droid.sh",
                "size_bytes": 1,
                "sha256": "sha256:" + "9" * 64,
            }
            if link in {"policy", "policy-diagnostic"}
            else None
        ),
        "bound_runtime_inputs": [
            {
                "relative_path": f"runtime_inputs/{name}",
                "size_bytes": (
                    bound_paths[name].stat().st_size if name in bound_paths else 1
                ),
                "sha256": (
                    _sha(bound_paths[name])
                    if name in bound_paths
                    else "sha256:" + "5" * 64
                ),
            }
            for name in bound_names
        ],
        "runtime_source_packet": {
            "receipt_digest": runtime_source_digest,
            "packet_sha256": "sha256:" + "6" * 64,
            "packet_size_bytes": 1,
            "install_roots": [],
            "runtime_dependency_wheels": [],
            "redistribution_permitted": True,
        },
        "runtime_entrypoint": "provider_runtime/run_adp_arena_provider_runtime.sh",
        "expected_output_filename": {
            "construction": "native_task_arena_construction_result.v1.json",
            "controls": "native_task_arena_control_result.v1.json",
            "policy": "native_task_arena_policy_result.v1.json",
            "policy-diagnostic": (
                "native_task_arena_policy_diagnostic_result.v1.json"
            ),
        }[link],
        "policy_candidate_id": (
            "pi05_droid" if link in {"policy", "policy-diagnostic"} else None
        ),
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "packet_bytes_mutated": False,
        "scene_reconstructed_by_bundle": False,
        "native_application_claimed": False,
        "retry_cap": 0,
        "provider_zero_required_after_return": True,
        "blockers": [],
        "input_digest": "",
    }
    if link in {"policy", "policy-diagnostic"}:
        bound_spec = json.loads(
            bound_paths[
                "native_task_arena_policy_execution_spec.v1.json"
            ].read_text(encoding="utf-8")
        )
        manifest.update(
            {
                "policy_execution_spec_digest": bound_spec[
                    "execution_spec_digest"
                ],
                "policy_execution_authority": bound_spec[
                    "execution_authority"
                ],
                "policy_rights_binding": bound_spec[
                    "candidate_rights_binding"
                ],
            }
        )
    manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")
    receipt = {
        **manifest,
        "bundle_path": str(bundle),
        "bundle_size_bytes": bundle.stat().st_size,
        "bundle_sha256": _sha(bundle),
    }
    receipt_path = root / "native_task_arena_provider_bundle_receipt.v1.json"
    write_json(receipt_path, receipt)
    return receipt_path


def _attempt_authority(
    path: Path,
    *,
    bundle_receipt: Path,
    predecessor: dict[str, Path],
) -> Path:
    bundle = json.loads(bundle_receipt.read_text(encoding="utf-8"))
    predecessor_authority = json.loads(
        predecessor["authority"].read_text(encoding="utf-8")
    )
    reconciliation_path = path.parent / "reconciliation.json"
    if not reconciliation_path.exists():
        write_json(reconciliation_path, {"fixture": "validated-by-ledger-contract-stub"})
    reconciliation_record = {
        **_record(reconciliation_path),
        "receipt_digest": "sha256:" + "8" * 64,
        "entry_count": 0,
        "total_cost_usd": 0.05,
    }
    prior_spend = {
        "prior_terminal_attempts": [{"result": _record(predecessor["result"])}],
        "reconciliation": reconciliation_record,
        "actual_total_usd": 0.05,
    }
    authority = {
        "schema_version": paid.AUTHORITY_SCHEMA_VERSION,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": "scene-840920-production-goal",
        "authorized_by": "nijelhunt_1",
        "authorized_on": "2026-08-16",
        "purpose": "one_shot_native_task_arena_execution",
        "provider": "vast",
        "paid_compute_authorized": True,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "zero_retry": True,
        "bundle_receipt": _record(bundle_receipt),
        "bundle_sha256": bundle["bundle_sha256"],
        "bundle_input_digest": bundle["input_digest"],
        "packet_receipt_digest": bundle["packet_receipt_digest"],
        "runtime_source_packet_receipt_digest": bundle["runtime_source_packet"][
            "receipt_digest"
        ],
        "execution_mode": bundle["execution_mode"],
        "policy_candidate_id": bundle["policy_candidate_id"],
        "blueprint_commit": COMMIT,
        "container_image": bundle["container_image"],
        "hard_attempt_spend_cap_usd": 2.0,
        "maximum_hourly_rate_usd": 1.0,
        "maximum_single_resource_ttl_seconds": 7_200,
        "aggregate_goal_spend_before_attempt_usd": 0.05,
        "aggregate_goal_spend_cap_usd": paid.AGGREGATE_GOAL_SPEND_CAP_USD,
        "prior_terminal_attempt": {
            "authority": _record(predecessor["authority"]),
            "terminal_result": _record(predecessor["result"]),
            "provider_zero": _record(predecessor["zero"]),
            "authority_digest": predecessor_authority["authorization_digest"],
            "attempt_cost_usd": 0.05,
            "actual_provider_charge_usd": 0.05,
        },
        "prior_terminal_attempts": prior_spend["prior_terminal_attempts"],
        "prior_spend_reconciliation": prior_spend["reconciliation"],
        "prior_actual_provider_spend_usd": 0.05,
        "active_instance_allowlist": {
            "external_provider_owned": [],
            "same_goal_concurrent": [],
        },
        "raw_nonredistributable_bytes_uploaded": False,
        "canonical_interiorgs_uploaded_or_mutated": False,
        "simulator_output_is_not_physical_evidence": True,
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    write_json(path, authority)
    return path


@pytest.fixture()
def lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    packet = tmp_path / "packet"
    packet.mkdir()
    packet_archive = packet / "native_task_arena_packet.zip"
    packet_archive.write_bytes(b"arena-packet")
    request = {
        "schema_version": "native_task_arena_packet_request.v1",
        "scene_id": SCENE_ID,
        "task_id": TASK_ID,
        "construction_bindings": {
            "schema_version": "paired_target_native_construction_bindings.v2"
        },
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    write_json(packet / "native_task_arena_packet_request.v1.json", request)
    # An executable plan with its assets really staged, because the profile
    # builder now asks the adapter whether it would accept this packet. A stub
    # that declares the executable schema but carries no objects, articulation,
    # or cameras is a packet the runtime refuses -- so a fixture shaped that way
    # would assert that unlaunchable packets get profiles.
    from tests.test_native_task_arena_runtime import (
        KINEMATIC_ARTICULATION_USDA,
        STATIC_COLLISION_USDA,
        _sealed_scene_plan,
    )

    assets = packet / "assets"
    assets.mkdir(exist_ok=True)
    staged = []
    # The articulation is real USD: the profile builder asks the adapter
    # whether it would accept this packet, and an asset it cannot open is a
    # refusal. Placeholder bytes would assert that unopenable assets launch.
    from blueprint_pipeline.native_task_arena_runtime import (
        author_grounded_articulation,
    )

    collision = assets / "collision.usd"
    # real static collision USD: the pre-spend gate opens it and refuses a
    # convex hull PhysX could not GPU-cook
    collision.write_text(STATIC_COLLISION_USDA, encoding="utf-8")
    staged.append(collision)
    sealed = packet / "sealed_task.usda"
    sealed.write_text(KINEMATIC_ARTICULATION_USDA, encoding="utf-8")
    task = assets / "task.usda"
    lane_adaptation = author_grounded_articulation(sealed, task)
    assert lane_adaptation is not None
    staged.append(task)
    scene_plan = _sealed_scene_plan()
    scene_plan["scene_id"] = SCENE_ID
    scene_plan["task_id"] = TASK_ID
    scene_plan["scenario"] = {"cell_id": "canonical.seed_1", "seed": 1}
    scene_plan["asset_directory"] = "assets"
    for row, path in zip(scene_plan["objects"], staged, strict=True):
        row["usd_path"] = f"assets/{path.name}"
        row["size_bytes"] = path.stat().st_size
        row["sha256"] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    for row in scene_plan["objects"]:
        if row.get("object_type") == "ARTICULATION":
            # the staged bytes carry the grounding; the plan declares the same
            row["articulation_adaptation"] = lane_adaptation
    scene_plan["plan_digest"] = canonical_digest(
        scene_plan, digest_field="plan_digest"
    )
    write_json(packet / "native_task_arena_scene_plan.v1.json", scene_plan)
    packet_receipt = {
        "schema_version": "native_task_arena_packet_receipt.v1",
        "status": "construction_packet_completed",
        "implementation_commit": COMMIT,
        "scene_id": SCENE_ID,
        "task_id": TASK_ID,
        "request_digest": request["request_digest"],
        "arena_scene_plan_digest": scene_plan["plan_digest"],
        "receipt_digest": "",
    }
    packet_receipt["receipt_digest"] = canonical_digest(
        packet_receipt, digest_field="receipt_digest"
    )
    write_json(packet / builder.PACKET_RECEIPT_NAME, packet_receipt)
    execution_admission = {
        "schema_version": "native_task_execution_admission.v1",
        "status": "admitted_for_native_gpu_construction",
        "scene_id": SCENE_ID,
        "task_id": TASK_ID,
        "asset_id": "task_asset",
        "registered_asset_sha256": "sha256:" + "1" * 64,
        "runtime_image": "nvcr.io/nvidia/isaac-sim:6.0.1@sha256:" + "2" * 64,
        "execution_candidate_digest": "sha256:" + "3" * 64,
        "collision_intent_digest": "sha256:" + "4" * 64,
        "native_runtime_result_digest": "sha256:" + "5" * 64,
        "packet_receipt_digest": packet_receipt["receipt_digest"],
        "scene_plan_digest": scene_plan["plan_digest"],
        "native_gpu_cooking_readback_qualified": True,
        "native_simulation_step_qualified": True,
        "construction_authorized": True,
        "controls_executed": False,
        "learned_policy_executed": False,
        "physical_equivalence_claimed": False,
        "receipt_digest": "",
    }
    execution_admission["receipt_digest"] = canonical_digest(
        execution_admission, digest_field="receipt_digest"
    )
    write_json(
        packet / "native_task_execution_admission.v1.json",
        execution_admission,
    )
    source_packet = tmp_path / "runtime_source_packet.json"
    runtime_source = {
        "schema_version": "native_task_runtime_source_packet.v1",
        "status": "ready",
        "receipt_digest": "",
    }
    runtime_source["receipt_digest"] = canonical_digest(
        runtime_source, digest_field="receipt_digest"
    )
    write_json(source_packet, runtime_source)
    predecessor = _predecessor(tmp_path / "predecessor")
    reconciled = {
        "prior_terminal_attempts": [{"result": _record(predecessor["result"])}],
        "reconciliation": None,
        "actual_total_usd": 0.05,
    }
    monkeypatch.setattr(
        paid, "validate_bound_lane_prior_spend", lambda *_args, **_kwargs: reconciled
    )
    construction = tmp_path / "construction_result.json"
    construction_value = {
        "schema_version": "native_task_arena_construction_result.v1",
        "status": "completed",
        "construction_gate_qualified": True,
        "scene_plan_digest": scene_plan["plan_digest"],
        "result_digest": "",
    }
    construction_value["result_digest"] = canonical_digest(
        construction_value, digest_field="result_digest"
    )
    write_json(construction, construction_value)
    control_plan_value = {
        "schema_version": "adp_task_control_plan.v1",
        "scene_plan_digest": scene_plan["plan_digest"],
        "construction_result_digest": construction_value["result_digest"],
        "candidate_policy_queried": False,
        "plan_digest": "",
    }
    control_plan_value["plan_digest"] = canonical_digest(
        control_plan_value, digest_field="plan_digest"
    )
    control_plan = tmp_path / "adp_task_control_plan.v1.json"
    write_json(control_plan, control_plan_value)
    control_execution_spec_value = {
        "schema_version": "adp_task_control_execution_spec.v1",
        "control_selection": "control_pair",
        "task_kind": scene_plan["task_kind"],
        "scene_plan_digest": scene_plan["plan_digest"],
        "construction_result_digest": construction_value["result_digest"],
        "control_plan_digest": control_plan_value["plan_digest"],
        "candidate_policy_queried": False,
        "prior_zero_action_result_digest": None,
        "execution_spec_digest": "",
    }
    control_execution_spec_value["execution_spec_digest"] = canonical_digest(
        control_execution_spec_value, digest_field="execution_spec_digest"
    )
    control_execution_spec = tmp_path / "adp_task_control_execution_spec.v1.json"
    write_json(control_execution_spec, control_execution_spec_value)
    control = tmp_path / "control_result.json"
    control_value = {
        "schema_version": "native_task_arena_control_result.v1",
        "status": "completed",
        "controls_qualified": True,
        "scene_plan_digest": scene_plan["plan_digest"],
        "construction_result_digest": construction_value["result_digest"],
        "control_pair": {
            "cell_id": "canonical.seed_1",
            "cell_admitted_for_policy_execution": True,
            "pair_digest": "sha256:" + "7" * 64,
        },
        "result_digest": "",
    }
    control_value["result_digest"] = canonical_digest(
        control_value, digest_field="result_digest"
    )
    write_json(control, control_value)
    diagnostic_control = tmp_path / "diagnostic_control_result.json"
    diagnostic_control_value = json.loads(json.dumps(control_value))
    diagnostic_control_value["status"] = "blocked"
    diagnostic_control_value["controls_qualified"] = False
    diagnostic_control_value["control_pair"].update(
        {
            "cell_admitted_for_policy_execution": False,
            "controls": [
                {
                    "control_id": "zero_action_negative",
                    "control_passed": True,
                    "observed_outcome": "never_moved",
                    "receipt_digest": "sha256:" + "8" * 64,
                }
            ],
        }
    )
    diagnostic_control_value["control_pair"]["pair_digest"] = canonical_digest(
        diagnostic_control_value["control_pair"], digest_field="pair_digest"
    )
    diagnostic_control_value["result_digest"] = canonical_digest(
        diagnostic_control_value, digest_field="result_digest"
    )
    write_json(diagnostic_control, diagnostic_control_value)
    policy, endpoint, identity = _candidate_runtime_binding("pi05_droid")
    policy_identity = asdict(policy)
    readiness = load_scene_policy_readiness(
        ADP009D_POLICY_READINESS_PATH,
        scenario_suite_path=ADP009D_SCENARIO_SUITE_PATH,
    )
    policy_spec = tmp_path / "policy_execution_spec.json"
    policy_value = {
        "schema_version": "native_task_arena_policy_execution_spec.v1",
        "candidate_id": "pi05_droid",
        "task_id": TASK_ID,
        "cell_id": "canonical.seed_1",
        "prompt": "Open the articulated fixture.",
        "scene_plan_digest": scene_plan["plan_digest"],
        "construction_result_digest": construction_value["result_digest"],
        "control_result_digest": control_value["result_digest"],
        "control_pair_digest": control_value["control_pair"]["pair_digest"],
        "policy_endpoint": endpoint,
        "policy_spec": policy_identity,
        "policy_identity_receipt": identity,
        "candidate_rights_binding": build_candidate_policy_rights(
            readiness,
            candidate_id="pi05_droid",
            policy_spec=policy_identity,
            runtime_robot_id=scene_plan["robot"]["robot_id"],
            scene_plan_digest=scene_plan["plan_digest"],
        ),
        "max_policy_queries": 56,
        "open_loop_horizon": policy.open_loop_horizon,
        "overview_camera_policy_input": False,
        "policy_may_grade_itself": False,
        "execution_authority": "qualified_controls_evaluation",
        "execution_spec_digest": "",
    }
    policy_value["execution_spec_digest"] = canonical_digest(
        policy_value, digest_field="execution_spec_digest"
    )
    write_json(policy_spec, policy_value)
    diagnostic_policy_spec = tmp_path / "policy_diagnostic_execution_spec.json"
    diagnostic_policy_value = json.loads(json.dumps(policy_value))
    diagnostic_policy_value.update(
        {
            "control_result_digest": diagnostic_control_value["result_digest"],
            "control_pair_digest": diagnostic_control_value["control_pair"]["pair_digest"],
            "execution_authority": (
                "development_only_unqualified_controls_canonical_diagnostic"
            ),
            "claim_ceiling": (
                "development_only_policy_motion_diagnostic_not_scoring_not_ranking_"
                "not_qualification"
            ),
            "initial_state": "canonical_scene_reset",
            "controls_qualified": False,
            "zero_action_negative_bound_separately": True,
            "scientific_scoring_permitted": False,
            "ranking_permitted": False,
            "qualification_permitted": False,
        }
    )
    diagnostic_policy_value["execution_spec_digest"] = canonical_digest(
        diagnostic_policy_value, digest_field="execution_spec_digest"
    )
    write_json(diagnostic_policy_spec, diagnostic_policy_value)
    bound_paths = {
        "adp009d_scene_840920_policy_readiness.v1.json": (
            ADP009D_POLICY_READINESS_PATH
        ),
        "third_scene_840920_task_a_scenario_suite.v1.json": (
            ADP009D_SCENARIO_SUITE_PATH
        ),
        "native_task_arena_construction_result.v1.json": construction,
        "adp_task_control_plan.v1.json": control_plan,
        "adp_task_control_execution_spec.v1.json": control_execution_spec,
        "native_task_arena_control_result.v1.json": control,
        "native_task_arena_policy_execution_spec.v1.json": policy_spec,
    }
    bundle_receipts = {}
    authorities = {}
    for link in builder.LINKS:
        link_bound_paths = dict(bound_paths)
        if link == "policy-diagnostic":
            link_bound_paths.update(
                {
                    "native_task_arena_control_result.v1.json": diagnostic_control,
                    "native_task_arena_policy_execution_spec.v1.json": (
                        diagnostic_policy_spec
                    ),
                }
            )
        bundle_receipt = _provider_bundle(
            tmp_path / f"{link}-bundle",
            link=link,
            packet_digest=packet_receipt["receipt_digest"],
            runtime_source_digest=runtime_source["receipt_digest"],
            request_digest=request["request_digest"],
            scene_plan_digest=scene_plan["plan_digest"],
            bound_paths=link_bound_paths,
        )
        bundle_receipts[link] = bundle_receipt
        authorities[link] = _attempt_authority(
            tmp_path / f"{link}-authority.json",
            bundle_receipt=bundle_receipt,
            predecessor=predecessor,
        )
    reconciliation_record = json.loads(
        authorities["construction"].read_text(encoding="utf-8")
    )["prior_spend_reconciliation"]
    reconciled["reconciliation"] = reconciliation_record
    monkeypatch.setattr(
        live_profile,
        "validate_same_goal_spend_reconciliation",
        lambda *_args, **_kwargs: ({"entries": []}, reconciliation_record),
    )
    return {
        "packet": packet,
        "bundle_receipts": bundle_receipts,
        "authorities": authorities,
        "source_packet": source_packet,
        "construction": construction,
        "control": control,
        "policy_spec": policy_spec,
        "diagnostic_control": diagnostic_control,
        "diagnostic_policy_spec": diagnostic_policy_spec,
    }


def _build(lane, link: str, **overrides):
    arguments = {
        "link": link,
        "packet_dir": lane["packet"],
        "bundle_receipt_path": lane["bundle_receipts"][link],
        "attempt_authority_path": lane["authorities"][link],
        "runtime_source_packet_path": lane["source_packet"],
        "source_commit": overrides.pop("source_commit", COMMIT),
        "raw_manifest_uri": URI,
        "expected_scene_id": SCENE_ID,
        "expected_task_id": TASK_ID,
    }
    if link in {"controls", "policy", "policy-diagnostic"}:
        arguments["construction_result_path"] = lane["construction"]
    if link in {"policy", "policy-diagnostic"}:
        diagnostic = link == "policy-diagnostic"
        arguments["control_result_path"] = lane[
            "diagnostic_control" if diagnostic else "control"
        ]
        arguments["policy_execution_spec_path"] = lane[
            "diagnostic_policy_spec" if diagnostic else "policy_spec"
        ]
    arguments.update(overrides)
    return builder.build_native_task_arena_live_profile(**arguments)


def test_policy_profile_exposes_private_digest_bound_native_policy(lane: dict) -> None:
    profile = _build(lane, "policy")
    binding = profile["native_policy_binding"]

    assert binding["candidate_id"] == "pi05_droid"
    assert binding["robot"]["runtime_robot_id"] == "franka_panda"
    assert binding["robot"]["rights_embodiment_id"] == "franka"
    assert binding["robot"]["alias_binding_digest"].startswith("sha256:")
    assert binding["task"]["task_id"] == TASK_ID
    assert binding["policy"]["action_adapter"]
    assert binding["runtime"]["arena_container_digest_pinned"] is True
    assert binding["runtime"]["candidate_policy_container"] is False
    assert "@sha256:" in binding["runtime"]["arena_container_image"]
    assert binding["rights"]["candidate_rights_binding_digest"].startswith(
        "sha256:"
    )


def test_policy_profile_exposes_campaign_and_makes_it_service_readable(
    lane: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_path = lane["packet"].parent / "policy-campaign.json"
    write_json(campaign_path, {"campaign": "bound"})
    campaign_binding = {
        "campaign": _record(campaign_path),
        "campaign_id": "scene-840920-policy-pair-1",
        "campaign_digest": "sha256:" + "9" * 64,
        "member_id": "pi05_droid",
        "launch_id": "launch-policy-pi05-campaign-1",
        "resource_name": "blueprint-native-task-policy-pi05-" + "a" * 32,
        "sibling_member_id": "groot_n17_droid",
        "sibling_launch_id": "launch-policy-groot-campaign-1",
        "sibling_resource_name": "blueprint-native-task-policy-groot-" + "b" * 32,
    }
    authority_path = lane["authorities"]["policy"]
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["policy_campaign_binding"] = campaign_binding
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    write_json(authority_path, authority)
    monkeypatch.setattr(
        paid,
        "_native_policy_campaign_binding",
        lambda **_kwargs: campaign_binding,
    )

    profile = _build(lane, "policy")

    assert profile["native_policy_binding"]["policy_campaign"] == {
        key: campaign_binding[key]
        for key in (
            "campaign_id",
            "campaign_digest",
            "member_id",
            "launch_id",
            "resource_name",
            "sibling_member_id",
            "sibling_launch_id",
            "sibling_resource_name",
        )
    }
    campaign_inputs = [
        row
        for row in profile["immutable_inputs"]
        if row["name"] == "native_task_arena_policy_campaign"
    ]
    assert campaign_inputs == [
        {
            "name": "native_task_arena_policy_campaign",
            "path": str(campaign_path.resolve()),
            "digest": _sha(campaign_path),
        }
    ]


def _preallocation_zero_fixture(
    root: Path, *, authority_path: Path, sibling: Path | None = None
) -> tuple[Path, set[Path]]:
    root.mkdir()
    original = root / "original.json"
    watchdog = root / "watchdog.json"
    cleanup = root / "cleanup.json"
    teardown = root / "teardown.json"
    api_zero = root / "api-provider-zero.json"
    for path, value in (
        (original, {"kind": "allocator-result"}),
        (watchdog, {"kind": "watchdog"}),
        (cleanup, {"kind": "cleanup"}),
        (teardown, {"kind": "teardown"}),
        (api_zero, {"kind": "api-provider-zero"}),
    ):
        write_json(path, value)
    terminal = root / "terminal-result.json"
    write_json(
        terminal,
        {
            "original_allocator_result": _record(original),
            "watchdog_handoff": _record(watchdog),
            "object_store_cleanup": _record(cleanup),
        },
    )
    zero = root / "provider-zero.json"
    write_json(
        zero,
        {
            "schema_version": "native_task_arena_provider_zero.v1",
            "status": "completed_preallocation_provider_zero",
            "attempt_authority": _record(authority_path),
            "teardown": _record(teardown),
            "terminal_result": _record(terminal),
            "watchdog": _record(watchdog),
            "object_store_cleanup": _record(cleanup),
            "api_provider_zero": _record(api_zero),
            "sibling_preallocation_closeouts": (
                [_record(sibling)] if sibling is not None else []
            ),
        },
    )
    return zero, {
        authority_path.resolve(),
        original.resolve(),
        watchdog.resolve(),
        cleanup.resolve(),
        teardown.resolve(),
        api_zero.resolve(),
        terminal.resolve(),
        zero.resolve(),
    }


def _pre_spend_zero_fixture(
    root: Path, *, authority_path: Path
) -> tuple[Path, set[Path]]:
    root.mkdir()
    original = root / "original.json"
    preflight = root / "preflight.json"
    consumption = root / "consumption.json"
    teardown = root / "teardown.json"
    api_zero = root / "api-provider-zero.json"
    for path, value in (
        (original, {"kind": "allocator-result"}),
        (preflight, {"kind": "pre-spend-preflight"}),
        (consumption, {"kind": "authority-consumption"}),
        (teardown, {"kind": "teardown"}),
        (api_zero, {"kind": "api-provider-zero"}),
    ):
        write_json(path, value)
    terminal = root / "terminal-result.json"
    write_json(
        terminal,
        {
            "closeout_kind": "pre_spend_preflight_blocked_before_allocation",
            "original_allocator_result": _record(original),
            "pre_spend_preflight": _record(preflight),
            "authority_consumption_record": _record(consumption),
        },
    )
    zero = root / "provider-zero.json"
    write_json(
        zero,
        {
            "schema_version": "native_task_arena_provider_zero.v1",
            "status": "completed_preallocation_provider_zero",
            "attempt_authority": _record(authority_path),
            "teardown": _record(teardown),
            "terminal_result": _record(terminal),
            "pre_spend_preflight": _record(preflight),
            "authority_consumption_record": _record(consumption),
            "api_provider_zero": _record(api_zero),
            "sibling_preallocation_closeouts": [],
        },
    )
    return zero, {
        authority_path.resolve(),
        original.resolve(),
        preflight.resolve(),
        consumption.resolve(),
        teardown.resolve(),
        api_zero.resolve(),
        terminal.resolve(),
        zero.resolve(),
    }


def test_policy_profile_retains_transitive_preallocation_closeout_permissions(
    lane: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority_path = lane["authorities"]["policy"]
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    closed_authority_path = Path(
        authority["prior_terminal_attempt"]["authority"]["path"]
    )
    sibling_zero, sibling_files = _preallocation_zero_fixture(
        tmp_path / "sibling-closeout", authority_path=closed_authority_path
    )
    primary_zero, primary_files = _preallocation_zero_fixture(
        tmp_path / "primary-closeout",
        authority_path=closed_authority_path,
        sibling=sibling_zero,
    )
    authority["prior_terminal_attempt"]["provider_zero"] = _record(primary_zero)
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    write_json(authority_path, authority)
    monkeypatch.setattr(
        builder, "validate_native_task_arena_paid_attempt_authority", lambda *_a, **_k: None
    )

    profile = _build(lane, "policy")

    retained = {Path(row["path"]).resolve() for row in profile["immutable_inputs"]}
    assert primary_files | sibling_files <= retained


def test_policy_profile_retains_pre_spend_closeout_permissions(
    lane: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority_path = lane["authorities"]["policy"]
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    closed_authority_path = Path(
        authority["prior_terminal_attempt"]["authority"]["path"]
    )
    provider_zero, closeout_files = _pre_spend_zero_fixture(
        tmp_path / "pre-spend-closeout", authority_path=closed_authority_path
    )
    authority["prior_terminal_attempt"]["provider_zero"] = _record(provider_zero)
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    write_json(authority_path, authority)
    monkeypatch.setattr(
        builder, "validate_native_task_arena_paid_attempt_authority", lambda *_a, **_k: None
    )

    profile = _build(lane, "policy")

    retained = {Path(row["path"]).resolve() for row in profile["immutable_inputs"]}
    assert closeout_files <= retained


def test_policy_profile_refuses_tampered_preallocation_closeout_binding(
    lane: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority_path = lane["authorities"]["policy"]
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    closed_authority_path = Path(
        authority["prior_terminal_attempt"]["authority"]["path"]
    )
    provider_zero, _ = _preallocation_zero_fixture(
        tmp_path / "preallocation-closeout", authority_path=closed_authority_path
    )
    zero = json.loads(provider_zero.read_text(encoding="utf-8"))
    zero["api_provider_zero"]["sha256"] = "sha256:" + "0" * 64
    write_json(provider_zero, zero)
    authority["prior_terminal_attempt"]["provider_zero"] = _record(provider_zero)
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    write_json(authority_path, authority)
    monkeypatch.setattr(
        builder, "validate_native_task_arena_paid_attempt_authority", lambda *_a, **_k: None
    )

    with pytest.raises(
        TaskEvaluationLaunchError,
        match="immutable_input_binding_invalid:prior_terminal_provider_zero_api_provider_zero",
    ):
        _build(lane, "policy")


@pytest.mark.parametrize(
    "link,probe_kind",
    [
        ("construction", "native-task-arena-construction"),
        ("controls", "native-task-arena-controls"),
        ("policy", "native-task-arena-policy"),
        ("policy-diagnostic", "native-task-arena-policy-diagnostic"),
    ],
)
def test_each_link_routes_its_own_probe_kind(lane, link: str, probe_kind: str) -> None:
    argv = _build(lane, link)["allocator"]["argv"]

    assert argv[argv.index("--probe-kind") + 1] == probe_kind
    assert "--native-task-arena-packet" in argv
    assert "--native-task-arena-runtime-source-packet" in argv
    assert argv[argv.index("--native-task-arena-bundle-receipt") + 1] == str(
        lane["bundle_receipts"][link].resolve()
    )
    assert argv[argv.index("--native-task-arena-attempt-authority") + 1] == str(
        lane["authorities"][link].resolve()
    )


def test_controls_profile_forwards_digest_bound_warm_retention(lane) -> None:
    authority_path = lane["authorities"]["controls"]
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["retain_warm_session"] = True
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    write_json(authority_path, authority)

    argv = _build(lane, "controls")["allocator"]["argv"]

    assert "--native-task-arena-retain-warm-session" in argv


def test_controls_profile_forwards_authorized_external_active_instances(lane) -> None:
    authority_path = lane["authorities"]["controls"]
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["active_instance_allowlist"]["external_provider_owned"] = [41, 42]
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    write_json(authority_path, authority)

    argv = _build(lane, "controls")["allocator"]["argv"]

    indices = [
        index
        for index, value in enumerate(argv)
        if value == "--adp-allowed-active-vast-instance-id"
    ]
    assert [argv[index + 1] for index in indices] == ["41", "42"]


def test_controls_profile_routes_small_bundle_to_retained_instance(lane) -> None:
    bundle_receipt = lane["bundle_receipts"]["controls"]
    bundle = json.loads(bundle_receipt.read_text(encoding="utf-8"))
    now = time.time()
    session = {
        "schema_version": warm_authority.SESSION_SCHEMA_VERSION,
        "generated_at": "fixed",
        "status": "ready",
        "provider": "vast",
        "instance_id": 123,
        "container_image": bundle["container_image"],
        "runtime_dependency_packet_sha256": bundle["runtime_source_packet"][
            "packet_sha256"
        ],
        "runtime_dependency_packet_size_bytes": bundle["runtime_source_packet"][
            "packet_size_bytes"
        ],
        "runtime_dependency_cache_ready": True,
        "ssh_host": "ssh.example",
        "ssh_port": 12345,
        "watchdog_pid": 456,
        "watchdog_deadline_epoch": now + 3600,
        "max_hourly_rate_usd": 0.8,
        "hard_cap_usd": 2.0,
        "continuing_spend": True,
        "raw_secret_values_recorded": False,
    }
    session["session_digest"] = warm_authority._session_digest(session)
    session_path = lane["packet"].parent / "warm-session.json"
    write_json(session_path, session)
    warm_authority_path = lane["packet"].parent / "warm-authority.json"
    warm_authority.materialize_native_task_arena_warm_attempt_authority(
        warm_session_path=session_path,
        bundle_receipt_path=bundle_receipt,
        prepared_bundle=bundle,
        authorization_reference="current production goal",
        authorized_by="user",
        authorized_on="2026-08-21",
        output_path=warm_authority_path,
        observed_now_epoch=now,
    )

    profile = _build(
        lane,
        "controls",
        attempt_authority_path=warm_authority_path,
        warm_session_path=session_path,
    )
    argv = profile["allocator"]["argv"]

    assert argv[argv.index("--native-task-arena-warm-session") + 1] == str(
        session_path.resolve()
    )
    assert argv[argv.index("--adp-allowed-active-vast-instance-id") + 1] == "123"
    assert "native_task_arena_warm_session" in {
        row["name"] for row in profile["immutable_inputs"]
    }


def test_default_budget_matches_the_attempt_authority(lane) -> None:
    profile = _build(lane, "construction")

    assert profile["allocator"]["max_spend_usd"] == 2.0


def test_profile_accepts_rate_above_cap_when_ttl_projection_fits(lane) -> None:
    authority_path = lane["authorities"]["policy-diagnostic"]
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority.update(
        {
            "maximum_hourly_rate_usd": 0.64,
            "hard_attempt_spend_cap_usd": 0.5,
            "maximum_single_resource_ttl_seconds": 2_800,
        }
    )
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    write_json(authority_path, authority)

    profile = _build(
        lane,
        "policy-diagnostic",
        max_hourly_rate_usd=0.64,
        max_spend_usd=0.5,
        hard_ttl_seconds=2_800,
    )

    assert profile["allocator"]["max_spend_usd"] == 0.5
    argv = profile["allocator"]["argv"]
    assert argv[argv.index("--adp-max-hourly-rate-usd") + 1] == "0.64"
    assert argv[argv.index("--adp-hard-ttl-seconds") + 1] == "2800"


def test_scene_and_task_are_part_of_the_profile_and_source_identity(lane) -> None:
    profile = _build(lane, "construction")

    assert SCENE_ID in profile["profile_id"]
    assert TASK_ID in profile["profile_id"]
    assert SCENE_ID in profile["source_bundle"]["bundle_id"]
    assert TASK_ID in profile["source_bundle"]["bundle_id"]


@pytest.mark.parametrize(
    "field,value,blocker",
    [
        ("expected_scene_id", "840313", "scene_identity_mismatch"),
        ("expected_task_id", "other_task", "task_identity_mismatch"),
    ],
)
def test_wrong_scene_or_task_is_refused_before_allocation(
    lane, field: str, value: str, blocker: str
) -> None:
    with pytest.raises(TaskEvaluationLaunchError, match=blocker):
        _build(lane, "construction", **{field: value})


def test_packet_request_identity_mutation_is_refused_before_allocation(lane) -> None:
    request_path = lane["packet"] / "native_task_arena_packet_request.v1.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["scene_id"] = "840313"
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    write_json(request_path, request)

    with pytest.raises(TaskEvaluationLaunchError, match="scene_identity_mismatch"):
        _build(lane, "construction")


def test_wrong_scene_plan_predecessor_is_refused_before_allocation(lane) -> None:
    construction = json.loads(lane["construction"].read_text(encoding="utf-8"))
    construction["scene_plan_digest"] = "sha256:" + "9" * 64
    construction["result_digest"] = canonical_digest(
        construction, digest_field="result_digest"
    )
    write_json(lane["construction"], construction)

    with pytest.raises(TaskEvaluationLaunchError, match="construction_result_invalid"):
        _build(lane, "controls")


def test_authority_budget_mismatch_is_refused_before_allocation(lane) -> None:
    with pytest.raises(TaskEvaluationLaunchError, match="attempt_authority_invalid"):
        _build(lane, "construction", max_spend_usd=1.5)


def test_bundle_receipt_and_packet_receipt_remain_distinct_predecessors(lane) -> None:
    profile = _build(lane, "construction")
    inputs = {row["name"]: row for row in profile["immutable_inputs"]}

    assert inputs["source_bundle_manifest"]["path"] == str(
        lane["bundle_receipts"]["construction"].resolve()
    )
    assert inputs["evaluation_run_spec"]["path"] == str(
        (lane["packet"] / builder.PACKET_RECEIPT_NAME).resolve()
    )
    assert inputs["native_task_arena_attempt_authority"]["path"] == str(
        lane["authorities"]["construction"].resolve()
    )
    assert inputs["native_task_execution_admission"]["path"] == str(
        (
            lane["packet"] / "native_task_execution_admission.v1.json"
        ).resolve()
    )
    assert inputs["source_bundle_manifest"] != inputs["evaluation_run_spec"]


def test_missing_execution_admission_is_refused_before_profile(lane) -> None:
    (lane["packet"] / "native_task_execution_admission.v1.json").unlink()

    with pytest.raises(TaskEvaluationLaunchError):
        _build(lane, "construction")


def test_provider_bundle_byte_tamper_is_refused_before_allocation(lane) -> None:
    receipt = json.loads(
        lane["bundle_receipts"]["construction"].read_text(encoding="utf-8")
    )
    Path(receipt["bundle_path"]).write_bytes(b"tampered-provider-bundle")

    with pytest.raises(TaskEvaluationLaunchError, match="provider_bundle_invalid"):
        _build(lane, "construction")


def test_policy_profile_refuses_bundle_rights_not_bound_to_external_spec(
    lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = lane["bundle_receipts"]["policy"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["policy_rights_binding"] = {
        **receipt["policy_rights_binding"],
        "rights_receipt_digest": "sha256:" + "0" * 64,
    }
    receipt["input_digest"] = canonical_digest(
        {
            key: value
            for key, value in receipt.items()
            if key not in {"bundle_path", "bundle_size_bytes", "bundle_sha256"}
        },
        digest_field="input_digest",
    )
    monkeypatch.setattr(
        builder,
        "BUNDLE_LOADERS",
        {
            **builder.BUNDLE_LOADERS,
            builder.POLICY_PROBE_KIND: lambda *_args, **_kwargs: receipt,
        },
    )

    with pytest.raises(
        TaskEvaluationLaunchError,
        match="native_task_arena_policy_execution_spec_invalid",
    ):
        _build(lane, "policy")


def test_permissive_attempt_authority_is_refused_before_allocation(lane) -> None:
    authority_path = lane["authorities"]["construction"]
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["maximum_provider_allocations"] = 2
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    write_json(authority_path, authority)

    with pytest.raises(TaskEvaluationLaunchError, match="attempt_authority_invalid"):
        _build(lane, "construction")


@pytest.mark.parametrize("link", ["construction", "controls", "policy"])
def test_the_shared_controls_are_present(lane, link: str) -> None:
    profile = _build(lane, link)

    assert profile["required_controls"]["provider_zero_required"] is True
    assert profile["required_controls"]["teardown_required"] is True
    assert profile["required_controls"]["webapp_status_sync_required"] is True
    assert profile["webapp_sync"] == {"max_attempts": 20}
    assert profile["allocator"]["retry_cap"] == 0
    assert sorted(profile["terminal_contract"]["required_path_fields"]) == [
        "artifact_manifest_path",
        "teardown_manifest_path",
    ]


def test_policy_profile_terminal_contract_rehearses_without_provider(
    lane, tmp_path: Path
) -> None:
    profile_path = tmp_path / "unpublished-policy-profile.json"
    write_json(profile_path, _build(lane, "policy"))

    receipt = rehearsal.rehearse_lane_terminal_contract(
        profile_path=profile_path,
        lane_module="native_task_arena_vast.py",
        lane="native-task-arena-policy",
    )

    assert receipt["status"] == "would_pass"
    assert receipt["blockers"] == []
    assert receipt["provider_mutation_performed"] is False
    assert receipt["paid_resource_allocated"] is False


def test_vast_geolocation_preference_is_digest_bound_to_the_profile(lane) -> None:
    profile = _build(
        lane,
        "controls",
        preferred_geolocation_regex="virginia|california|oregon|texas",
    )

    assert profile["runtime_environment"] == {
        "BLUEPRINT_VAST_PREFERRED_GEOLOCATION_REGEX": (
            "virginia|california|oregon|texas"
        )
    }
    assert profile["profile_digest"] == canonical_digest(
        profile, digest_field="profile_digest"
    )


@pytest.mark.parametrize(
    "link,omitted",
    [
        ("controls", "construction_result_path"),
        ("policy", "control_result_path"),
        ("policy", "policy_execution_spec_path"),
    ],
)
def test_a_link_without_its_predecessor_is_refused_before_allocation(
    lane, link: str, omitted: str
) -> None:
    """The allocator names the same absence, but only after it has rented."""

    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane, link, **{omitted: None})

    assert "native_task_arena_predecessor_required" in str(excinfo.value)


def test_construction_needs_no_predecessor(lane) -> None:
    """It is the head of the chain; demanding one would deadlock it."""

    assert _build(lane, "construction")["allocator"]["argv"]


@pytest.mark.parametrize("ttl", [900, 20_000], ids=["under-band", "over-band"])
def test_a_ttl_outside_the_allocator_band_is_refused_here(lane, ttl: int) -> None:
    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane, "construction", hard_ttl_seconds=ttl)

    assert "hard_ttl_out_of_band" in str(excinfo.value)


def test_a_packet_from_another_commit_is_refused(lane) -> None:
    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane, "construction", source_commit="b" * 40)

    assert "bundle_commit_not_source_commit" in str(excinfo.value)


def test_each_predecessor_result_is_pinned_by_digest(lane) -> None:
    """This link's verdict is only about the packet it actually consumed."""

    inputs = {row["name"]: row for row in _build(lane, "policy")["immutable_inputs"]}

    for name, path in (
        ("native_task_arena_construction_result", lane["construction"]),
        ("native_task_arena_control_result", lane["control"]),
    ):
        expected = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        assert inputs[name]["digest"] == expected


def test_the_allocator_still_demands_the_predecessors_this_builder_carries() -> None:
    """Guards the ordering itself, not just this profile's rendering."""

    source = (
        REPO_ROOT / "src" / "blueprint_pipeline" / "paid_resource_allocator.py"
    ).read_text(encoding="utf-8")

    assert "native_task_arena_construction_result" in source
    assert "native_task_arena_control_result" in source
    assert "native_task_arena_policy_execution_spec" in source


def test_no_profile_is_built_for_a_packet_the_runtime_would_refuse(lane) -> None:
    """The chokepoint: no profile, no authority consumed, no provider.

    Two paid attempts were spent learning things the adapter's own pre-build
    checks answer for free -- a payload that was not USD, then a contact sensor
    whose logical id the runtime did not admit. None of those checks need
    Isaac, so the profile builder asks the adapter directly, and a packet it
    would refuse never becomes launchable.
    """

    packet = lane["packet"]
    plan_path = packet / "native_task_arena_scene_plan.v1.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["articulation"]["contact_sensors"][0]["logical_sensor_id"] = "not_a_channel"
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    write_json(plan_path, plan)

    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane, "construction")

    message = str(excinfo.value)
    assert "native_task_arena_runtime_would_refuse" in message
    assert "contact_sensor_contract_invalid:0" in message


def test_a_scene_plan_that_cannot_be_read_blocks_rather_than_passes(lane) -> None:
    """An unreadable plan is refused, never treated as acceptable."""

    plan_path = lane["packet"] / "native_task_arena_scene_plan.v1.json"
    plan_path.write_text("{ not json", encoding="utf-8")

    with pytest.raises(TaskEvaluationLaunchError) as excinfo:
        _build(lane, "construction")

    assert "native_task_arena_scene_plan" in str(excinfo.value)



def test_every_link_defaults_to_the_durable_machine_avoidlist() -> None:
    """A machine proven bad by a paid run must not be relearned by the next one.

    The adapter already records a host whose container never reaches its
    onstart heartbeat, but its default avoidlist lives under the per-launch job
    root, so each launch started blind. On 2026-08-25 Vast machine 144209 took
    three consecutive GR00T launches that way -- GPU allocated, container gone,
    torn down -- while the same bundle had completed a full episode elsewhere.

    Defaulting the shared link parser to one durable path under the launch
    state root makes the exclusion cumulative across launches.
    """

    source = (
        REPO_ROOT / "scripts" / "build_native_task_arena_live_profile.py"
    ).read_text(encoding="utf-8")

    # One declaration, shared by construction/controls/policy/policy-diagnostic.
    assert source.count('target.add_argument(\n            "--machine-avoidlist"') == 1
    assert "default=DEFAULT_MACHINE_AVOIDLIST_PATH," in source

    durable = builder.DEFAULT_MACHINE_AVOIDLIST_PATH
    assert durable.endswith(
        "/task-evaluation-launch-runs/vast_machine_avoidlist.json"
    )
    # Durable means outside any single run: the state root itself, never a
    # launch-scoped or attempt-scoped subdirectory.
    assert "/attempts/" not in durable
    assert "-codex" not in durable
