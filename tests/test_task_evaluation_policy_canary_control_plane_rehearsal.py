"""Hermetic rehearsal of the policy canary control plane, intake to paid boundary.

The paired policy canary crosses six production workers before a GPU exists:
Website intake, the launch preparation worker, the episode compilation worker,
the configured-controls progression timer, the launch activation worker, and
the policy canary dispatcher.  Until this test existed every one of those
boundaries was pinned only by a hand-written fixture of the neighbouring
stage's artifact.  On 2026-09-01 two boundary drifts reached production, each
costing a full no-spend replay to discover: the progression CLI forwarded a
canary-only queue root into ``advance_configured_controls_plan`` (a
``TypeError`` that exited the timer unit), and the activation gate demanded a
qualified native construction result while the compiled controls-pending
lineage carried the episode compilation result the progression worker really
binds.  PR #1535 removed the equivalent loop for the GPU-side worker with a
hermetic lifecycle rehearsal; this module is the control-plane counterpart.

Every stage below is the real production function.  Each stage consumes the
files the previous stage sealed, so a contract drift between neighbours fails
here in seconds instead of after a 40 minute production replay.  Only object
storage, the release-window publisher, the native episode compiler, the
provider bundle builder, and the paid allocator are replaced, and each
replacement keeps the production shape the next real validator enforces.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import os
import pwd
import zipfile
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import (
    task_evaluation_configured_controls_progression_worker as progression_worker,
)
from blueprint_pipeline import task_evaluation_policy_canary_dispatcher as dispatcher
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_episode_compilation_worker import (
    process_episode_compilation_queue,
)
from blueprint_pipeline.task_evaluation_launch_activation_worker import (
    process_launch_activation_queue,
)
from blueprint_pipeline.task_evaluation_launch_preparation_worker import (
    collect_preparation_references,
    process_launch_preparation_queue,
)
from blueprint_pipeline.task_evaluation_native_arena_episode_compiler import (
    OUTPUT_SCHEMA_VERSION as COMPILER_OUTPUT_SCHEMA_VERSION,
)
from blueprint_pipeline.task_evaluation_native_arena_preparation_adapter import (
    RESULT_SCHEMA_VERSION as ADAPTER_RESULT_SCHEMA_VERSION,
)
from blueprint_pipeline.task_evaluation_policy_canary_preparation_dispatch import (
    maybe_dispatch_policy_canary_preparation,
)
from blueprint_pipeline.task_evaluation_policy_canary_scene_setup import (
    materialize_policy_canary_presubmission_setup,
)
from scripts.attach_internal_policy_canary_setup import (
    materialize_policy_canary_launch_profile,
)
from tests.test_task_evaluation_launch_dispatcher import (
    _profile as base_launch_profile,
    _request as base_launch_request,
)
from tests.test_task_evaluation_launch_preparation_contract import (
    request as preparation_contract_request,
)
from tests.test_task_evaluation_launch_preparation_worker import (
    request_with_fetchable_bytes,
)
from tests.test_task_evaluation_policy_canary_scene_setup import (
    COMMIT,
    _kwargs as presubmission_kwargs,
)


SERVICE_ACCOUNT = pwd.getpwuid(os.geteuid()).pw_name
TEAM_NAMESPACE = "blueprint-internal"
CANARY_PREFIX = "s3://blueprint/policy-canary/"
PRODUCTION_INPUT_PREFIX = "s3://blueprint/task-evaluation/production-inputs/"
RELEASE_WINDOW_PREFIX = PRODUCTION_INPUT_PREFIX + "coordinator-release-windows/"
ALLOWED_PREFIXES = [
    "s3://blueprint-production-inputs/",
    CANARY_PREFIX,
    PRODUCTION_INPUT_PREFIX,
]
#: The fields the activation gate reads from ``lineage.construction_result``
#: for a controls-pending canary.  The profile authors that reference from the
#: predecessor run's sealed compilation result; the rehearsal pins that the
#: shape the gate accepts is the shape the real compilation worker seals.
COMPILATION_LINEAGE_SHAPE = (
    "schema_version",
    "status",
    "configured_scene_revision_digest",
    "provider_mutation_performed",
    "paid_execution_requested",
    "blockers",
)


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
    value[field] = canonical_digest(value, digest_field=field)
    return value


class _ObjectStore:
    """Byte-exact stand-in for the object store every worker fetches from."""

    def __init__(self) -> None:
        self.payloads: dict[str, bytes] = {}
        self.fetched: list[str] = []

    def seal(self, uri: str, payload: bytes) -> dict[str, Any]:
        self.payloads[uri] = payload
        return {"uri": uri, "digest": _sha(payload), "size_bytes": len(payload)}

    def seal_json(self, uri: str, value: Any) -> dict[str, Any]:
        return self.seal(uri, json.dumps(value, sort_keys=True).encode())

    def fetch(self, uri: str, destination: Path, maximum_bytes: int) -> None:
        if uri not in self.payloads:
            raise AssertionError(f"rehearsal object store has no bytes for {uri}")
        payload = self.payloads[uri]
        assert len(payload) <= maximum_bytes
        self.fetched.append(uri)
        destination.write_bytes(payload)

    def publish_release_window(self, *, path: Path, object_name: str) -> dict[str, Any]:
        payload = path.read_bytes()
        reference = self.seal(RELEASE_WINDOW_PREFIX + object_name, payload)
        return {
            **reference,
            "full_byte_service_account_readback_passed": True,
            "readback_digest": reference["digest"],
            "readback_size_bytes": reference["size_bytes"],
        }


def _fetchable_configured_scene(store: _ObjectStore) -> tuple[dict[str, Any], str]:
    """The configured Scene 839873 revision with every reference fetchable."""

    value = preparation_contract_request()
    value["team_namespace"] = TEAM_NAMESPACE
    value["expected_production_commit"] = COMMIT
    value, payloads = request_with_fetchable_bytes(value)
    store.payloads.update(payloads)
    # The activation worker parses the revision's source manifest and rights
    # admission as JSON claims; the shared fixture fills them with opaque
    # bytes, so re-seal the revision around sealed claims.
    revision_reference = value["scene"]["configured_revision"]
    revision = json.loads(store.payloads[revision_reference["uri"]])
    scene_id = str(value["scene"]["identity"]["id"])
    for field, schema, status, digest_field in (
        ("manifest", "task_evaluation_scene_source_manifest.v1", "retained", "source_manifest_digest"),
        ("rights_admission", "task_evaluation_scene_rights_admission.v1", "admitted", "rights_admission_digest"),
    ):
        claim = _sealed(
            {"schema_version": schema, "status": status, "scene_id": scene_id, digest_field: ""},
            digest_field,
        )
        reference = revision["source"][field]
        reference.update(store.seal_json(reference["uri"], claim))
    revision["revision_digest"] = ""
    _sealed(revision, "revision_digest")
    revision_reference.update(store.seal_json(revision_reference["uri"], revision))
    value["task"]["configured_scene_revision_digest"] = revision["revision_digest"]
    fields = (
        "scene",
        "construction",
        "robot",
        "controller",
        "task",
        "sensors",
        "runtime",
        "execution_adapter",
        "spend",
    )
    return {field: value[field] for field in fields}, value["task"][
        "configured_scene_revision_digest"
    ]


def _release_window_template() -> dict[str, Any]:
    return _sealed(
        {
            "schema_version": (
                "task_evaluation_configured_controls_release_window_template.v1"
            ),
            "status": "authorized_for_dynamic_release",
            "team_namespace": TEAM_NAMESPACE,
            "expected_production_commit": COMMIT,
            "allowed_mutations": [
                "profile_publication",
                "catalog_synchronization",
                "standing_authorization",
            ],
            "provider_allowlist": ["vast"],
            "maximum_hard_cap_usd": 4.0,
            "valid_for_seconds": 3600,
            "released_by": "blueprint-policy-lead",
            "release_reference": "automatic policy-canary activation",
            "provider_resource_allocation_allowed": False,
            "paid_request_allowed": False,
            "template_digest": "",
        },
        "template_digest",
    )


def _predecessor_compilation_result(revision_digest: str) -> dict[str, Any]:
    return _sealed(
        {
            "schema_version": "task_evaluation_episode_compilation_result.v1",
            "status": "compiled_for_production_launch",
            "configured_scene_revision_digest": revision_digest,
            "provider_mutation_performed": False,
            "paid_execution_requested": False,
            "blockers": [],
            "result_digest": "",
        },
        "result_digest",
    )


def _website_launch(
    tmp_path: Path, store: _ObjectStore
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    """Publish the canary profile exactly as production does and submit it."""

    configured_root = tmp_path / "configured-scene"
    kwargs = presubmission_kwargs(configured_root)
    for field in ("activation_digest", "capture_session_id", "intake_id"):
        kwargs.pop(field)
    configured_preparation, revision_digest = _fetchable_configured_scene(store)
    progression_path = _write(
        configured_root / "configured-progression.json",
        {
            "configured_scene_revision_digest": revision_digest,
            "episode_preparation_request": configured_preparation,
            "episode_preparation_request_digest": canonical_digest(
                configured_preparation
            ),
        },
    )
    base_profile = base_launch_profile(configured_root)
    base_profile["profile_id"] = "scene839873-configured-r4"
    base_profile["source_commit"] = COMMIT
    _sealed(base_profile, "profile_digest")
    base_profile_path = _write(configured_root / "base-launch-profile.json", base_profile)
    predecessor = _predecessor_compilation_result(revision_digest)
    lineage = {
        "kind": "predecessor",
        **{
            name: store.seal_json(
                f"{CANARY_PREFIX}activation/{name.replace('_', '-')}.json",
                {"schema_version": f"rehearsal_{name}.v1", "status": "terminal"},
            )
            for name in (
                "prior_authority",
                "prior_result",
                "prior_launch_receipt",
                "prior_webapp_sync",
                "prior_provider_zero",
                "prior_spend_reconciliation",
            )
        },
        "construction_result": store.seal_json(
            f"{CANARY_PREFIX}activation/construction-result.json", predecessor
        ),
    }
    runtime_bundle = zipfile_bytes({"runtime/README": b"rehearsal runtime source\n"})
    kwargs.update(
        {
            "profile_id": "scene839873-internal-policy-canary-current",
            "configured_source_commit": COMMIT,
            "configured_offering_configuration_run_id": "scene839873-configuration",
            "offering_digest": "sha256:" + "f" * 64,
            "scene_revision_digest": revision_digest,
            "launch_profile_path": base_profile_path,
            "configured_progression_path": progression_path,
            "policy_controller_configuration": store.seal_json(
                f"{CANARY_PREFIX}controller.json", {"controller": "paired-policy"}
            ),
            "native_controller_configuration": store.seal_json(
                f"{CANARY_PREFIX}native-controller.json", {"controller": "native"}
            ),
            "runtime_source_bundle": store.seal(
                f"{CANARY_PREFIX}runtime-source.zip", runtime_bundle
            ),
            "runtime_source_implementation_commit": COMMIT,
            "model_rights": store.seal_json(
                f"{CANARY_PREFIX}model-rights.json", {"rights": "recorded"}
            ),
            "activation_release_window_template": store.seal_json(
                f"{CANARY_PREFIX}activation/release-window-template.json",
                _release_window_template(),
            ),
            "activation_lineage": lineage,
            "hard_ttl_seconds": 9_000,
            "output_dir": tmp_path / "presubmission",
        }
    )
    emitted = materialize_policy_canary_presubmission_setup(**kwargs)
    wrapper = emitted["profile_materialization_input"]
    public = emitted["setup"]
    profile = materialize_policy_canary_launch_profile(
        base_configured_profile=base_profile,
        profile_materialization_input=wrapper,
    )
    resource = wrapper["internal_policy_canary_execution_plan"]["resource_authority"]
    matrix = public["episode_presets"][0]["matrix"]
    request = base_launch_request(profile)
    request.update(
        {
            "source_launch_id": public["source_launch_id"],
            "offering_digest": public["offering_digest"],
            "setup_digest": public["setup_digest"],
            "preset_id": "quick_10",
            "run_kind": "internal_policy_canary",
            "claim_ceiling": "diagnostic_policy_execution",
            "scene_revision_digest": public["scene_revision_digest"],
            "scene_controls_status_at_submission": "configured_controls_pending",
            "team_namespace": TEAM_NAMESPACE,
            "robot_preset_id": "droid_franka_panda_robotiq_2f85_v1",
            "policy_candidate_ids": ["pi05_droid", "groot_n17_droid"],
            "episode_plan": {
                "preset": "quick_10",
                "episodes_per_policy": 10,
                "policy_count": 2,
                "learned_policy_rollout_count": 20,
                "variation_matrix_digest": matrix["matrix_digest"],
                "resolved_cells": copy.deepcopy(matrix["cells"]),
                "resolved_seeds": [cell["seed"] for cell in matrix["cells"]],
                "coverage_gaps": [],
                "diagnostic_control_rollouts": {
                    "zero_action_count": 10,
                    "deterministic_scripted_positive_count": 10,
                    "total_count": 20,
                    "blocking_for_policy_execution": False,
                },
            },
            "notification": {
                "email": "robotics@example.com",
                "notify_on": ["completed", "blocked", "cancelled"],
            },
            "authorization": {
                "actor": {"id": "robotics-member", "role": "team_member"},
                "authorized_at": "2026-09-01T16:30:00Z",
                "spend": {
                    "approved": True,
                    "currency": "USD",
                    "max_spend_usd": resource["hard_cap_usd"],
                    "hard_ttl_seconds": resource["hard_ttl_seconds"],
                },
                "execution": {"approved": True},
            },
            "required_controls": {
                **profile["required_controls"],
                "maximum_provider_allocations": 1,
                "retry_cap": 0,
            },
            "controls_qualification_bypassed": False,
            "scene_promotion_permitted": False,
            "official_ranking_permitted": False,
        }
    )
    _sealed(request, "request_digest")
    return profile, request, Path(emitted["execution_setup_template_path"]), predecessor


def zipfile_bytes(members: dict[str, bytes]) -> bytes:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def _production_shaped_episode_compiler(
    *, envelope: dict[str, Any], materialized_references: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    """Write the exact artifact layout the real compiler seals, without Isaac.

    The compilation worker validates this output with the same
    ``_validated_compiler_output`` gate production applies to the real
    compiler, and the activation worker then validates the adapter result and
    packet root through its real episode-compilation branch.
    """

    assert "scene.configured_revision.configured_scene_bundle" in materialized_references
    root = Path(output_root)
    packet_zip = root / "native-task-arena-bundle.zip"
    packet_zip.write_bytes(
        zipfile_bytes({"native_task_arena_packet_request.v1.json": b"{}\n"})
    )
    adapter_dir = root / "native-arena-adapter"
    packet_root = adapter_dir / "construction-packet"
    packet_receipt = _write(
        packet_root / "native_task_arena_packet_receipt.v1.json",
        {
            "schema_version": "native_task_arena_packet_receipt.v1",
            "scene_id": "interiorgs-839873",
            "task_id": "scene-839873-mug-planar-push",
        },
    )
    _write(
        packet_root / "native_task_runtime_contract.v1.json",
        {
            "schema_version": "native_task_runtime_contract.v1",
            "robot": {"robot_id": "franka_panda"},
            "task_spec_digest": "sha256:" + "5" * 64,
        },
    )
    runtime_receipt = _write(
        adapter_dir / "runtime-source" / "native_task_runtime_source_packet.v1.json",
        {
            "schema_version": "native_task_runtime_source_packet.v1",
            "packet_sha256": "sha256:" + "d" * 64,
            "packet_size_bytes": 4_287_162_924,
            "redistribution_permitted": True,
        },
    )
    adapter = _sealed(
        {
            "schema_version": ADAPTER_RESULT_SCHEMA_VERSION,
            "status": "native_arena_adapter_materialized",
            "adapter_kind": "native_task_arena",
            "adapter_version": "v1",
            "preparation_id": envelope["preparation_id"],
            "source_commit": envelope["expected_production_commit"],
            "packet_root": str(packet_root),
            "runtime_source_receipt": str(runtime_receipt),
            "packet_receipt_digest": _sha(packet_receipt.read_bytes()),
            "runtime_source_receipt_digest": _sha(runtime_receipt.read_bytes()),
            "result_digest": "",
        },
        "result_digest",
    )
    adapter_path = _write(
        adapter_dir / "task_evaluation_native_arena_adapter_result.v1.json", adapter
    )
    packet_bytes = packet_zip.read_bytes()
    return _sealed(
        {
            "schema_version": COMPILER_OUTPUT_SCHEMA_VERSION,
            "status": "completed",
            "run_id": envelope["run_id"],
            "configured_scene_revision_digest": envelope[
                "configured_scene_revision_digest"
            ],
            "compiled_episode_packet": {
                "format": "native_task_arena_bundle_zip",
                "path": str(packet_zip),
                "digest": _sha(packet_bytes),
                "size_bytes": len(packet_bytes),
            },
            "adapter_result": {
                "path": str(adapter_path),
                "digest": adapter["result_digest"],
                "packet_receipt_digest": adapter["packet_receipt_digest"],
                "runtime_source_receipt_digest": adapter[
                    "runtime_source_receipt_digest"
                ],
            },
            "compiled_by_production": True,
            "customer_supplied_prebuilt_episode_packet": False,
            "provider_mutation_performed": False,
            "paid_execution_requested": False,
            "raw_secret_values_recorded": False,
            "compiler_output_digest": "",
        },
        "compiler_output_digest",
    )


def test_policy_canary_control_plane_reaches_the_paid_boundary_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _ObjectStore()
    profile, request, execution_template, predecessor = _website_launch(tmp_path, store)
    launch_state = tmp_path / "launch-state"
    preparation_queue = tmp_path / "preparation-queue"
    preparation_inputs = tmp_path / "preparation-inputs"
    compilation_queue = tmp_path / "episode-compilation-queue"
    compiled_outputs = tmp_path / "compiled-episodes"
    activation_queue = tmp_path / "activation-queue"
    activation_root = tmp_path / "activations"
    dispatch_queue = tmp_path / "policy-canary-dispatch-queue"
    for state in ("pending", "processing", "completed", "blocked"):
        (dispatch_queue / state).mkdir(parents=True)

    # Stage 1: the authenticated Website submission diverts to no-spend
    # preparation and seals the launch request, profile, and receipt.
    receipt = maybe_dispatch_policy_canary_preparation(
        request=request,
        profile=profile,
        blockers=[],
        state_root=launch_state,
        preparation_queue_root=preparation_queue,
    )
    assert receipt is not None
    assert receipt["status"] == "queued_for_no_spend_preparation", receipt["blockers"]
    assert receipt["allocator_invoked"] is False
    preparation_id = receipt["preparation_queue"]["preparation_id"]
    pending = next((preparation_queue / "pending").glob("*.json"))
    queued_request = json.loads(pending.read_text(encoding="utf-8"))["request"]
    expected_references = collect_preparation_references(queued_request)
    assert {row["uri"] for row in expected_references} <= set(store.payloads)

    # Stage 2: the preparation worker fetches and reads back every immutable
    # reference, then hands the configured scene to the compilation queue.
    prepared = process_launch_preparation_queue(
        queue_root=preparation_queue,
        input_root=preparation_inputs,
        allowed_uri_prefixes=ALLOWED_PREFIXES,
        service_account=SERVICE_ACCOUNT,
        source_commit=COMMIT,
        fetcher=store.fetch,
        episode_compilation_queue_root=compilation_queue,
    )
    preparation = prepared["results"][0]
    assert preparation["status"] == "queued_for_production_episode_compilation", (
        preparation.get("blockers")
    )
    assert preparation["full_byte_service_account_readback_passed"] is True
    assert preparation["episode_compilation_id"] == preparation_id
    assert prepared["provider_mutation_performed"] is False

    # Stage 3: the compilation worker compiles the no-spend packet.
    compiled = process_episode_compilation_queue(
        queue_root=compilation_queue,
        input_root=preparation_inputs,
        output_root=compiled_outputs,
        source_commit=COMMIT,
        episode_compiler=_production_shaped_episode_compiler,
    )
    compilation = compiled["results"][0]
    assert compilation["status"] == "compiled_for_production_launch", (
        compilation.get("blockers")
    )
    assert compilation["compiled_by_production"] is True
    assert {key: compilation[key] for key in COMPILATION_LINEAGE_SHAPE} == {
        key: predecessor[key] for key in COMPILATION_LINEAGE_SHAPE
    }

    # Stage 4: the progression timer tick.  A configured-controls plan is
    # present so the real ``advance_configured_controls_plan`` signature is
    # exercised beside the canary branch; before PR #1538 the canary-only
    # compilation root leaked into it as an unexpected keyword.
    plan_root = tmp_path / "plans"
    _write(plan_root / "configured-controls.json", {"schema_version": "test.plan.v1"})
    monkeypatch.setattr(
        progression_worker,
        "advance_policy_canary_activation",
        functools.partial(
            progression_worker.advance_policy_canary_activation,
            release_window_publisher_factory=lambda: store.publish_release_window,
        ),
    )
    tick = progression_worker.process_plans(
        plan_root=plan_root,
        autostart_intent_root=tmp_path / "autostart-intents",
        launch_state_root=launch_state,
        progression_root=tmp_path / "progression",
        preparation_queue_root=preparation_queue,
        episode_compilation_queue_root=compilation_queue,
        activation_queue_root=activation_queue,
        repo_root=tmp_path,
        webapp_secret_file=tmp_path / "webapp-secret",
        webapp_endpoint="https://tryblueprint.io/api/internal/task-evaluation-launch-submissions",
    )
    statuses = {row.get("status") for row in tick["rows"]}
    assert "policy_canary_activation_queued" in statuses, tick
    assert "blocked" in statuses, tick
    activation_pending = list((activation_queue / "pending").glob("*.json"))
    assert len(activation_pending) == 1
    activation_request = json.loads(activation_pending[0].read_text(encoding="utf-8"))[
        "request"
    ]
    assert activation_request["run_kind"] == "internal_policy_canary"
    assert activation_request["release_window"]["uri"].startswith(RELEASE_WINDOW_PREFIX)

    # Stage 5: the activation worker re-verifies the whole lineage and seals
    # the paired campaign queue plus the canary-only dispatch envelope.
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    authorization_dir = tmp_path / "standing-authorizations"
    authorization_dir.mkdir()
    activated = process_launch_activation_queue(
        queue_root=activation_queue,
        preparation_queue_root=preparation_queue,
        preparation_input_root=preparation_inputs,
        episode_compilation_queue_root=compilation_queue,
        episode_compilation_output_root=compiled_outputs,
        activation_root=activation_root,
        allowed_uri_prefixes=ALLOWED_PREFIXES,
        service_account=SERVICE_ACCOUNT,
        service_group=pwd.getpwuid(os.geteuid()).pw_name,
        repository_root=tmp_path,
        destination_prefix=PRODUCTION_INPUT_PREFIX + "task-evaluation-activations",
        release_window_prefix=RELEASE_WINDOW_PREFIX,
        profile_dir=profile_dir,
        webapp_catalog=tmp_path / "catalog.json",
        standing_authorization_dir=authorization_dir,
        policy_canary_dispatch_queue_root=dispatch_queue,
        source_commit=COMMIT,
        fetcher=store.fetch,
    )
    activation = activated["results"][0]
    assert activation["status"] == "policy_campaign_queue_materialized_no_execution", (
        activation.get("blockers")
    )
    assert activation["campaign_unit_count"] == 10
    assert activation["provider_mutation_performed"] is False
    assert activation["paid_execution_requested"] is False
    assert activation["profile_publication_performed"] is False
    runtime_inputs = json.loads(
        Path(activation["policy_canary_runtime_inputs_path"]).read_text(encoding="utf-8")
    )
    assert runtime_inputs["candidate_ids"] == ["pi05_droid", "groot_n17_droid"]
    assert len(runtime_inputs["cells"]) == 10
    assert all(
        cell["control_diagnostic"]["typed_gap"] == "controls_pending_at_submission"
        for cell in runtime_inputs["cells"]
    )
    assert runtime_inputs["resource_authority"]["hard_ttl_seconds"] == 9_000
    envelopes = list((dispatch_queue / "pending").glob("*.json"))
    assert len(envelopes) == 1

    # Stage 6: the dispatcher materializes the execution setup from the
    # published template, seals the session authority, builds the bundle, and
    # invokes the canonical allocator without ``--execute``.
    observed: dict[str, Any] = {}

    def fake_bundle(**kwargs: Any) -> dict[str, Any]:
        observed["bundle"] = kwargs
        job = Path(kwargs["job_dir"])
        job.mkdir(parents=True, exist_ok=True)
        bundle_path = job / "bundle.zip"
        bundle_path.write_bytes(zipfile_bytes({"session/README": b"rehearsal\n"}))
        receipt = {
            "bundle_sha256": _sha(bundle_path.read_bytes()),
            "bundle_path": str(bundle_path),
        }
        _write(
            job / "native_task_arena_policy_canary_session_bundle_receipt.v1.json",
            receipt,
        )
        return receipt

    def fake_allocator(argv: list[str]) -> int:
        observed["argv"] = list(argv)
        adapter = Path(argv[argv.index("--adapter-output") + 1])
        _write(adapter, {"status": "dry_run_ready", "provider_mutations_performed": 0})
        return 0

    monkeypatch.setattr(dispatcher, "build_policy_canary_session_bundle", fake_bundle)
    monkeypatch.setattr(dispatcher, "_default_allocator_runner", fake_allocator)
    setups = tmp_path / "execution-setups"
    setups.mkdir()
    dispatched = dispatcher.process_policy_canary_dispatch_queue(
        dispatch_queue_root=dispatch_queue,
        execution_setup_root=setups,
        execution_setup_template_path=execution_template,
        dispatch_root=tmp_path / "dispatches",
        implementation_commit=COMMIT,
        execute=False,
    )
    dispatch = dispatched["results"][0]
    assert dispatch["status"] == "prepared_no_execution", dispatch
    assert dispatch["retry_cap"] == 0
    assert dispatch["provider_mutation_performed"] is False
    assert (dispatch_queue / "completed" / envelopes[0].name).is_file()
    argv = observed["argv"]
    assert argv[:3] == ["gpu-canary", "--provider", "vast"]
    assert "--execute" not in argv
    assert argv[argv.index("--adp-max-spend-usd") + 1] == "4.0"
    assert argv[argv.index("--adp-hard-ttl-seconds") + 1] == "9000"
    assert Path(observed["bundle"]["session_authority_path"]).is_file()
    assert Path(observed["bundle"]["packet_dir"]).is_dir()
    setup = json.loads(
        next(setups.rglob("task_evaluation_policy_canary_execution_setup.v1.json")).read_text(
            encoding="utf-8"
        )
    )
    assert setup["status"] == "verified_runnable"
    assert setup["activation_digest"] == activation["policy_campaign_activation_digest"]
    assert setup["request_digest"] == request["request_digest"]
    assert store.fetched, "every immutable reference must reach the workers through the store"
