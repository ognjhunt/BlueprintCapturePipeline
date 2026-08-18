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
from pathlib import Path

import pytest

from blueprint_pipeline.adp009d_native_microcheck_bundle import (
    DEFAULT_IMAGE as QUALIFIED_ADP_IMAGE,
)
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
import blueprint_pipeline.native_task_arena_paid_authority as paid
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
    bundle.write_bytes(f"production-shaped-{link}-bundle".encode())
    mode = {
        "construction": "construction_canary",
        "controls": "controls",
        "policy": "policy",
    }[link]
    bound_names = {
        "construction": (),
        "controls": (
            "native_task_arena_construction_result.v1.json",
            "adp_task_control_plan.v1.json",
        ),
        "policy": (
            "native_task_arena_construction_result.v1.json",
            "native_task_arena_control_result.v1.json",
            "native_task_arena_policy_execution_spec.v1.json",
        ),
    }[link]
    manifest = {
        "schema_version": "native_task_arena_provider_bundle.v1",
        "generated_at": "2026-08-16T00:00:00+00:00",
        "status": "ready",
        "program_id": "arm-decision-proof-v1",
        "execution_mode": mode,
        "implementation_commit": COMMIT,
        "container_image": QUALIFIED_ADP_IMAGE,
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
        }[link],
        "policy_candidate_id": "pi05_droid" if link == "policy" else None,
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
        "aggregate_goal_spend_cap_usd": 12.0,
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
    policy_spec = tmp_path / "policy_execution_spec.json"
    policy_value = {
        "schema_version": "native_task_arena_policy_execution_spec.v1",
        "task_id": TASK_ID,
        "cell_id": "canonical.seed_1",
        "scene_plan_digest": scene_plan["plan_digest"],
        "construction_result_digest": construction_value["result_digest"],
        "control_result_digest": control_value["result_digest"],
        "control_pair_digest": control_value["control_pair"]["pair_digest"],
        "execution_spec_digest": "",
    }
    policy_value["execution_spec_digest"] = canonical_digest(
        policy_value, digest_field="execution_spec_digest"
    )
    write_json(policy_spec, policy_value)
    bound_paths = {
        "native_task_arena_construction_result.v1.json": construction,
        "native_task_arena_control_result.v1.json": control,
        "native_task_arena_policy_execution_spec.v1.json": policy_spec,
    }
    bundle_receipts = {}
    authorities = {}
    for link in builder.LINKS:
        bundle_receipt = _provider_bundle(
            tmp_path / f"{link}-bundle",
            link=link,
            packet_digest=packet_receipt["receipt_digest"],
            runtime_source_digest=runtime_source["receipt_digest"],
            request_digest=request["request_digest"],
            scene_plan_digest=scene_plan["plan_digest"],
            bound_paths=bound_paths,
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
    if link in {"controls", "policy"}:
        arguments["construction_result_path"] = lane["construction"]
    if link == "policy":
        arguments["control_result_path"] = lane["control"]
        arguments["policy_execution_spec_path"] = lane["policy_spec"]
    arguments.update(overrides)
    return builder.build_native_task_arena_live_profile(**arguments)


@pytest.mark.parametrize(
    "link,probe_kind",
    [
        ("construction", "native-task-arena-construction"),
        ("controls", "native-task-arena-controls"),
        ("policy", "native-task-arena-policy"),
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


def test_default_budget_matches_the_attempt_authority(lane) -> None:
    profile = _build(lane, "construction")

    assert profile["allocator"]["max_spend_usd"] == 2.0


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
    assert inputs["source_bundle_manifest"] != inputs["evaluation_run_spec"]


def test_provider_bundle_byte_tamper_is_refused_before_allocation(lane) -> None:
    receipt = json.loads(
        lane["bundle_receipts"]["construction"].read_text(encoding="utf-8")
    )
    Path(receipt["bundle_path"]).write_bytes(b"tampered-provider-bundle")

    with pytest.raises(TaskEvaluationLaunchError, match="provider_bundle_invalid"):
        _build(lane, "construction")


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
    assert profile["allocator"]["retry_cap"] == 0
    assert sorted(profile["terminal_contract"]["required_path_fields"]) == [
        "artifact_manifest_path",
        "teardown_manifest_path",
    ]


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
