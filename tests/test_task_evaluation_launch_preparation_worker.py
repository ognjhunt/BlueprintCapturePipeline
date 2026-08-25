from __future__ import annotations

import hashlib
import json
import os
import pwd
import sys
import types
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_preparation_queue import (
    launch_preparation_status,
    stage_launch_preparation_request,
)
from blueprint_pipeline.task_evaluation_launch_preparation_worker import (
    TaskEvaluationLaunchPreparationWorkerError,
    collect_preparation_references,
    default_reference_fetcher,
    main,
    materialize_preparation_references,
    process_launch_preparation_queue,
    validate_allowed_uri_prefixes,
)
from blueprint_pipeline.task_evaluation_scene_construction_recipe import (
    CAPABILITY_ORDER,
    SCHEMA_VERSION as CONSTRUCTION_RECIPE_SCHEMA_VERSION,
)
from tests.test_task_evaluation_launch_preparation_contract import (
    request,
    test_configuration_request,
)


SERVICE_ACCOUNT = pwd.getpwuid(os.geteuid()).pw_name


def request_with_fetchable_bytes(
    value: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, bytes]]:
    value = value or request()
    payloads: dict[str, bytes] = {}

    def replace(node) -> None:
        if isinstance(node, dict):
            if set(node) == {"uri", "digest", "size_bytes"}:
                payload = ("payload-for-" + node["uri"]).encode()
                node["digest"] = "sha256:" + hashlib.sha256(payload).hexdigest()
                node["size_bytes"] = len(payload)
                payloads[node["uri"]] = payload
                return
            for child in node.values():
                replace(child)
        elif isinstance(node, list):
            for child in node:
                replace(child)

    replace(value)
    if value.get("run_mode") == "episode_evaluation":
        def configured_ref(index: int) -> dict[str, object]:
            return {
                "uri": (
                    "s3://blueprint-production-inputs/configured-scene/"
                    f"object-{index}.json"
                ),
                "digest": f"sha256:{index:064x}",
                "size_bytes": 1000 + index,
            }

        configured_scene_bundle_bytes = b"configured-scene-bundle-bytes"
        configured_scene_bundle = {
            "uri": (
                "s3://blueprint-production-inputs/configured-scene-bundle.zip"
            ),
            "digest": "sha256:"
            + hashlib.sha256(configured_scene_bundle_bytes).hexdigest(),
            "size_bytes": len(configured_scene_bundle_bytes),
        }
        payloads[configured_scene_bundle["uri"]] = configured_scene_bundle_bytes
        configured_revision: dict[str, object] = {
            "schema_version": "task_evaluation_configured_scene_revision.v1",
            "status": "configured",
            "configuration_run_id": "scene-configuration-v1",
            "team_namespace": value["team_namespace"],
            "scene_identity": value["scene"]["identity"],
            "source_commit": value["expected_production_commit"],
            "source": {
                "manifest": configured_ref(1),
                "rights_admission": configured_ref(2),
                "rights_evidence": [
                    {
                        "role": "publisher_terms",
                        "artifact": configured_ref(25),
                    },
                    {
                        "role": "human_authority_record",
                        "artifact": configured_ref(26),
                    },
                ],
                "raw_source_sent_to_external_provider": False,
            },
            "appearance": {
                "observed_source": configured_ref(3),
                "object_removal_result": configured_ref(4),
                "configured_representation": configured_ref(5),
                "appearance_truth_source": "interiorgs_observed_plus_labeled_generated_edit",
            },
            "geometry": {
                "candidate_collision_source": configured_ref(6),
                "object_excision_result": configured_ref(7),
                "configured_collision": configured_ref(8),
                "validation": configured_ref(9),
                "observed_source_truth_claimed": False,
            },
            "replacement": {
                "identity": value["task"]["subject"]["identity"],
                "source_object": configured_ref(10),
                "asset": configured_ref(11),
                "static_qualification": configured_ref(12),
                "native_import_qualification": configured_ref(13),
                "physics_authority": "qualified_replacement_asset",
            },
            "registration": {
                "metric": configured_ref(14),
                "support_plane": configured_ref(15),
                "robot_mount_interface": configured_ref(16),
                "camera_calibration": configured_ref(17),
                "workspace_clearance": configured_ref(18),
            },
            "configured_scene_bundle": configured_scene_bundle,
            "task_template": {
                "identity": value["task"]["identity"],
                "definition": configured_ref(19),
                "success_criteria": configured_ref(20),
                "execution": configured_ref(21),
            },
            "robot_team_interface": {
                "scene_construction_repeated_per_evaluation": False,
                "configuration_run_executed_episode": False,
                "configuration_run_purpose": (
                    "build_and_publish_reusable_robot_neutral_scene"
                ),
                "episode_run_purpose": (
                    "evaluate_one_robot_or_policy_against_configured_scene"
                ),
                "episode_packet_compiled_by_production": True,
                "team_supplied_components": [
                    "robot_configuration",
                    "kinematics_and_joint_bounds",
                    "robot_to_scene_registration",
                    "controller_or_policy",
                    "camera_and_sensor_configuration",
                    "task_binding",
                    "episode_runtime",
                ],
                "configured_scene_components": [
                    "appearance",
                    "collision_geometry",
                    "replacement_assets",
                    "metric_registration",
                    "support_plane",
                    "robot_mount_interface",
                    "workspace_clearance",
                    "scene_camera_calibration",
                    "rights_and_provenance",
                    "task_templates",
                    "configured_scene_bundle",
                ],
                "production_route": (
                    "authenticated_webapp_to_task_evaluation_dispatcher"
                ),
            },
            "publication": {
                "bundle_manifest": configured_ref(22),
                "receipt": configured_ref(23),
                "full_byte_service_account_readback_passed": True,
            },
            "evaluation_admission": {
                "zero_action_required": True,
                "scripted_positive_required": True,
                "learned_policy_admitted": False,
            },
            "revision_digest": "",
        }
        for label, reference in (
            ("source-manifest", configured_revision["source"]["manifest"]),
            (
                "rights-admission",
                configured_revision["source"]["rights_admission"],
            ),
            *[
                (
                    f"rights-evidence-{index}",
                    evidence["artifact"],
                )
                for index, evidence in enumerate(
                    configured_revision["source"]["rights_evidence"]
                )
            ],
        ):
            payload = f"configured-revision-{label}".encode()
            reference["digest"] = (
                "sha256:" + hashlib.sha256(payload).hexdigest()
            )
            reference["size_bytes"] = len(payload)
            payloads[reference["uri"]] = payload
        configured_revision["revision_digest"] = canonical_digest(
            configured_revision, digest_field="revision_digest"
        )
        value["task"]["configured_scene_revision_digest"] = configured_revision[
            "revision_digest"
        ]
        revision_bytes = json.dumps(configured_revision, sort_keys=True).encode()
        revision_reference = value["scene"]["configured_revision"]
        revision_reference["digest"] = (
            "sha256:" + hashlib.sha256(revision_bytes).hexdigest()
        )
        revision_reference["size_bytes"] = len(revision_bytes)
        payloads[revision_reference["uri"]] = revision_bytes
    return value, payloads


def production_request_with_fetchable_bytes() -> tuple[dict[str, object], dict[str, bytes]]:
    value = test_configuration_request()
    value["construction"]["output_identity"] = {
        "id": "constructed-packet",
        "version": "v1",
    }
    value, payloads = request_with_fetchable_bytes(value)
    stages = []
    for index, capability in enumerate(CAPABILITY_ORDER):
        configuration_uri = (
            f"s3://blueprint-production-inputs/stage-{index + 1}.json"
        )
        configuration_bytes = json.dumps(
            {"stage": index + 1, "capability": capability}, sort_keys=True
        ).encode()
        payloads[configuration_uri] = configuration_bytes
        stages.append(
            {
                "stage_id": f"stage-{index + 1}",
                "capability": capability,
                "adapter": {"id": f"adapter-{index + 1}", "version": "v1"},
                "execution_class": "gpu_canary" if index in {0, 2, 4} else "no_spend",
                "configuration": {
                    "uri": configuration_uri,
                    "digest": "sha256:"
                    + hashlib.sha256(configuration_bytes).hexdigest(),
                    "size_bytes": len(configuration_bytes),
                },
                "depends_on": [] if index == 0 else [f"stage-{index}"],
            }
        )
    recipe: dict[str, object] = {
        "schema_version": CONSTRUCTION_RECIPE_SCHEMA_VERSION,
        "recipe_id": "source-object-replacement-v1",
        "team_namespace": value["team_namespace"],
        "scene_identity": value["scene"]["identity"],
        "task_identity": value["task"]["identity"],
        "subject_identity": value["task"]["subject"]["identity"],
        "source_manifest_digest": value["scene"]["source_manifest"]["digest"],
        "rights_admission_digest": value["scene"]["rights"]["admission"]["digest"],
        "stage_sequence": stages,
        "output_identity": value["construction"]["output_identity"],
        "provider_disclosure": {
            "raw_source_bytes_to_external_provider": False,
            "derived_runtime_processing_allowed": True,
            "provider_training_allowed": False,
            "public_redistribution_allowed": False,
        },
        "recipe_digest": "",
    }
    recipe["recipe_digest"] = canonical_digest(recipe, digest_field="recipe_digest")
    recipe_bytes = json.dumps(recipe, sort_keys=True).encode()
    recipe_ref = value["construction"]["recipe"]
    recipe_ref["digest"] = "sha256:" + hashlib.sha256(recipe_bytes).hexdigest()
    recipe_ref["size_bytes"] = len(recipe_bytes)
    payloads[recipe_ref["uri"]] = recipe_bytes
    return value, payloads


def fetcher(payloads: dict[str, bytes]):
    def fetch(uri: str, destination: Path, maximum_bytes: int) -> None:
        assert len(payloads[uri]) <= maximum_bytes
        destination.write_bytes(payloads[uri])

    return fetch


def fake_adapter(**kwargs) -> dict[str, object]:
    assert Path(kwargs["construction_bundle_path"]).is_file()
    assert Path(kwargs["runtime_source_bundle_path"]).is_file()
    result = {
        "status": "native_arena_adapter_materialized",
        "adapter_kind": "native_task_arena",
        "adapter_version": "v1",
        "packet_receipt_digest": "sha256:" + "c" * 64,
        "runtime_source_receipt_digest": "sha256:" + "d" * 64,
        "result_digest": "sha256:" + "e" * 64,
    }
    return result


def test_materializes_every_reference_and_full_byte_reads_back(tmp_path) -> None:
    value, payloads = request_with_fetchable_bytes()
    result = materialize_preparation_references(
        request=value,
        input_root=tmp_path / "inputs",
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT,
        source_commit=value["expected_production_commit"],
        fetcher=fetcher(payloads),
    )
    assert result["status"] == "inputs_materialized_awaiting_construction_adapter"
    # The configured-scene bundle is discovered only after the revision itself
    # is read back and is materialized by the queue worker in the next step.
    assert result["reference_count"] == len(
        collect_preparation_references(value)
    )
    assert result["full_byte_service_account_readback_passed"] is True
    assert result["provider_mutation_performed"] is False
    assert result["catalog_mutation_performed"] is False
    assert result["result_digest"] == canonical_digest(
        result, digest_field="result_digest"
    )
    for row in result["references"]:
        path = Path(row["materialized_path"])
        assert path.read_bytes() == payloads[row["uri"]]
        assert path.stat().st_mode & 0o777 == 0o440


def test_worker_claims_queue_and_seals_terminal_no_spend_result(tmp_path) -> None:
    value, payloads = request_with_fetchable_bytes()
    queue = tmp_path / "queue"
    intake = stage_launch_preparation_request(
        value=value, queue_root=queue, submitted_by="blueprint-webapp"
    )
    run = process_launch_preparation_queue(
        queue_root=queue,
        input_root=tmp_path / "inputs",
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT,
        source_commit=value["expected_production_commit"],
        fetcher=fetcher(payloads),
        adapter_materializer=fake_adapter,
        episode_compilation_queue_root=tmp_path / "episode-compilation",
    )
    assert run["status"] == "processed"
    assert run["processed_count"] == 1
    assert run["provider_mutation_performed"] is False
    assert launch_preparation_status(
        preparation_id=value["preparation_id"], queue_root=queue
    ) == {
        "schema_version": "task_evaluation_launch_preparation_status.v1",
        "status": "materialized",
        "preparation_id": value["preparation_id"],
        "run_mode": "episode_evaluation",
        "run_id": value["run_id"],
        "team_namespace": value["team_namespace"],
        "expected_production_commit": value["expected_production_commit"],
        "request_digest": intake["request_digest"],
        "provider_mutation_performed_by_status_read": False,
        "worker_status": "queued_for_production_episode_compilation",
        "source_commit": value["expected_production_commit"],
        "result_digest": run["results"][0]["result_digest"],
        "reference_count": len(payloads),
        "full_byte_service_account_readback_passed": True,
        "blockers": [],
        "provider_mutation_performed_by_worker": False,
        "catalog_mutation_performed_by_worker": False,
        "paid_execution_requested": False,
        "automatic_progression_required": True,
        "configured_scene_revision_digest": run["results"][0][
            "configured_scene_revision_digest"
        ],
        "configured_scene_bundle_digest": run["results"][0][
            "configured_scene_bundle_digest"
        ],
        "episode_compilation_id": value["preparation_id"],
        "episode_compilation_queue_envelope_digest": run["results"][0][
            "episode_compilation_queue_envelope_digest"
        ],
    }
    result_files = list((queue / "results").glob("*.json"))
    assert len(result_files) == 1
    sealed = json.loads(result_files[0].read_text())
    assert sealed["result_digest"] == canonical_digest(
        sealed, digest_field="result_digest"
    )
    compilation_files = list(
        (tmp_path / "episode-compilation" / "pending").glob("*.json")
    )
    assert len(compilation_files) == 1
    compilation = json.loads(compilation_files[0].read_text())
    assert compilation["materialized_references"] == run["results"][0][
        "references"
    ]
    assert compilation["production_compiler_owns_episode_packet"] is True
    assert compilation["customer_supplied_prebuilt_episode_packet"] is False


def test_episode_evaluation_blocks_configured_scene_bundle_readback_mismatch(
    tmp_path,
) -> None:
    value, payloads = request_with_fetchable_bytes()
    revision_ref = value["scene"]["configured_revision"]
    revision = json.loads(payloads[revision_ref["uri"]])
    revision["configured_scene_bundle"]["digest"] = "sha256:" + "f" * 64
    revision["revision_digest"] = canonical_digest(
        revision, digest_field="revision_digest"
    )
    value["task"]["configured_scene_revision_digest"] = revision[
        "revision_digest"
    ]
    revision_bytes = json.dumps(revision, sort_keys=True).encode()
    payloads[revision_ref["uri"]] = revision_bytes
    revision_ref["digest"] = "sha256:" + hashlib.sha256(revision_bytes).hexdigest()
    revision_ref["size_bytes"] = len(revision_bytes)
    queue = tmp_path / "queue"
    stage_launch_preparation_request(
        value=value, queue_root=queue, submitted_by="blueprint-webapp"
    )

    def forbidden_adapter(**_kwargs):
        raise AssertionError("scene revision mismatch must block before adapter")

    run = process_launch_preparation_queue(
        queue_root=queue,
        input_root=tmp_path / "inputs",
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT,
        source_commit=value["expected_production_commit"],
        fetcher=fetcher(payloads),
        adapter_materializer=forbidden_adapter,
        episode_compilation_queue_root=tmp_path / "episode-compilation",
    )

    assert run["results"][0]["status"] == "blocked"
    assert run["results"][0]["blockers"] == [
        "launch_preparation_reference_readback_mismatch"
    ]


def test_worker_accepts_recipe_without_prebuilt_packet_or_adapter_call(tmp_path) -> None:
    value, payloads = production_request_with_fetchable_bytes()
    queue = tmp_path / "queue"
    stage_launch_preparation_request(
        value=value, queue_root=queue, submitted_by="blueprint-webapp"
    )

    def forbidden_adapter(**_kwargs):
        raise AssertionError("production construction starts after preparation")

    run = process_launch_preparation_queue(
        queue_root=queue,
        input_root=tmp_path / "inputs",
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT,
        source_commit=value["expected_production_commit"],
        fetcher=fetcher(payloads),
        adapter_materializer=forbidden_adapter,
        construction_queue_root=tmp_path / "construction-queue",
    )

    result = run["results"][0]
    assert result["status"] == "queued_for_production_scene_configuration"
    assert result["run_mode"] == "scene_configuration"
    assert result["automatic_progression_required"] is True
    queued = list((tmp_path / "construction-queue" / "pending").glob("*.json"))
    assert len(queued) == 1
    envelope = json.loads(queued[0].read_text())
    assert envelope["run_id"] == value["run_id"]
    assert envelope["recipe_digest"] == result["construction_recipe_digest"]
    assert envelope["automatic_progression_required"] is True
    assert result["construction_packet_materialized"] is False
    assert result["construction_recipe_digest"].startswith("sha256:")
    assert result["construction_stage_configuration_count"] == 6
    assert result["construction_stage_configurations_readback_passed"] is True
    assert all(
        Path(row["materialized_path"]).read_bytes() == payloads[row["uri"]]
        for row in result["references"]
    )
    assert result["provider_mutation_performed"] is False
    assert result["paid_execution_requested"] is False


def test_worker_blocks_recipe_stage_configuration_outside_allowed_prefix(
    tmp_path,
) -> None:
    value, payloads = production_request_with_fetchable_bytes()
    recipe_uri = value["construction"]["recipe"]["uri"]
    recipe = json.loads(payloads[recipe_uri])
    recipe["stage_sequence"][2]["configuration"]["uri"] = (
        "s3://unapproved-team/stage-3.json"
    )
    recipe["recipe_digest"] = canonical_digest(
        recipe, digest_field="recipe_digest"
    )
    recipe_bytes = json.dumps(recipe, sort_keys=True).encode()
    payloads[recipe_uri] = recipe_bytes
    value["construction"]["recipe"]["digest"] = (
        "sha256:" + hashlib.sha256(recipe_bytes).hexdigest()
    )
    value["construction"]["recipe"]["size_bytes"] = len(recipe_bytes)
    queue = tmp_path / "queue"
    stage_launch_preparation_request(
        value=value, queue_root=queue, submitted_by="blueprint-webapp"
    )

    run = process_launch_preparation_queue(
        queue_root=queue,
        input_root=tmp_path / "inputs",
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT,
        source_commit=value["expected_production_commit"],
        fetcher=fetcher(payloads),
        adapter_materializer=fake_adapter,
        construction_queue_root=tmp_path / "construction-queue",
    )

    assert run["results"][0]["status"] == "blocked"
    assert run["results"][0]["blockers"] == [
        "launch_preparation_reference_prefix_not_allowed"
    ]


def test_worker_blocks_unapproved_storage_prefix_before_fetch(tmp_path) -> None:
    value, payloads = request_with_fetchable_bytes()
    queue = tmp_path / "queue"
    stage_launch_preparation_request(
        value=value, queue_root=queue, submitted_by="blueprint-webapp"
    )

    def forbidden_fetch(*_args):
        raise AssertionError("unapproved prefix must block before fetch")

    run = process_launch_preparation_queue(
        queue_root=queue,
        input_root=tmp_path / "inputs",
        allowed_uri_prefixes=["s3://different-team/"],
        service_account=SERVICE_ACCOUNT,
        source_commit=value["expected_production_commit"],
        fetcher=forbidden_fetch,
        adapter_materializer=fake_adapter,
    )
    assert run["results"][0]["status"] == "blocked"
    assert run["results"][0]["blockers"] == [
        "launch_preparation_reference_prefix_not_allowed"
    ]
    assert run["results"][0]["preparation_id"] == value["preparation_id"]
    assert launch_preparation_status(
        preparation_id=value["preparation_id"], queue_root=queue
    )["status"] == "blocked"


def test_worker_cli_fails_closed_without_production_configuration(capsys) -> None:
    assert main([]) == 2
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "blocked"
    assert receipt["blockers"] == [
        "launch_preparation_worker_configuration_invalid"
    ]
    assert receipt["provider_mutation_performed"] is False


@pytest.mark.parametrize(
    "prefix",
    [
        "s3://blueprint-production-inputs",
        "s3://user@example.test/team/",
        "file:///var/lib/blueprint/",
        "https://example.test/team/?token=secret",
    ],
)
def test_operator_storage_prefixes_fail_closed_at_directory_boundaries(
    prefix: str,
) -> None:
    with pytest.raises(
        TaskEvaluationLaunchPreparationWorkerError,
        match="launch_preparation_allowed_uri_prefix_invalid",
    ):
        validate_allowed_uri_prefixes([prefix])


def test_s3_fetch_uses_canonical_private_secret_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = {
        "BLUEPRINT_WAM_OBJECT_STORE_ACCESS_KEY_ID_FILE": "access",
        "BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY_FILE": "secret",
        "BLUEPRINT_WAM_OBJECT_STORE_ENDPOINT_URL_FILE": "https://objects.example.test",
        "BLUEPRINT_WAM_OBJECT_STORE_REGION_FILE": "region-1",
    }
    for environment_name, value in values.items():
        path = tmp_path / environment_name.lower()
        path.write_text(value + "\n", encoding="utf-8")
        path.chmod(0o600)
        monkeypatch.setenv(environment_name, str(path))

    observed: dict[str, object] = {}

    class Body:
        def __init__(self) -> None:
            self.payload = bytearray(b"payload")

        def read(self, count: int) -> bytes:
            chunk = bytes(self.payload[:count])
            del self.payload[:count]
            return chunk

    class Client:
        def get_object(self, **kwargs):
            observed["request"] = kwargs
            return {"ContentLength": 7, "Body": Body()}

    def client(name: str, **kwargs):
        observed["name"] = name
        observed["kwargs"] = kwargs
        return Client()

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=client))
    destination = tmp_path / "download"
    default_reference_fetcher(
        "s3://blueprint-production-inputs/object.bin", destination, 7
    )

    assert destination.read_bytes() == b"payload"
    assert observed == {
        "name": "s3",
        "kwargs": {
            "aws_access_key_id": "access",
            "aws_secret_access_key": "secret",
            "endpoint_url": "https://objects.example.test",
            "region_name": "region-1",
        },
        "request": {
            "Bucket": "blueprint-production-inputs",
            "Key": "object.bin",
        },
    }


def test_existing_different_result_is_preserved_and_conflict_is_sealed(
    tmp_path: Path,
) -> None:
    value, payloads = request_with_fetchable_bytes()
    queue = tmp_path / "queue"
    receipt = stage_launch_preparation_request(
        value=value, queue_root=queue, submitted_by="blueprint-webapp"
    )
    queue_name = Path(receipt["queue_path"]).name
    results = queue / "results"
    results.mkdir(mode=0o750)
    existing = {
        "schema_version": "task_evaluation_launch_preparation_result.v1",
        "status": "blocked",
        "preparation_id": value["preparation_id"],
        "blockers": ["prior-immutable-result"],
        "result_digest": "",
    }
    existing["result_digest"] = canonical_digest(
        existing, digest_field="result_digest"
    )
    (results / queue_name).write_text(
        json.dumps(existing, sort_keys=True) + "\n", encoding="utf-8"
    )

    run = process_launch_preparation_queue(
        queue_root=queue,
        input_root=tmp_path / "inputs",
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT,
        source_commit=value["expected_production_commit"],
        fetcher=fetcher(payloads),
        adapter_materializer=fake_adapter,
    )

    assert json.loads((results / queue_name).read_text()) == existing
    assert run["results"][0]["status"] == "blocked"
    assert run["results"][0]["blockers"] == [
        "launch_preparation_immutable_result_conflict"
    ]
    conflicts = list((results / "conflicts").glob("*.json"))
    assert len(conflicts) == 1
    status = launch_preparation_status(
        preparation_id=value["preparation_id"], queue_root=queue
    )
    assert status["status"] == "blocked"
    assert status["worker_status"] == "blocked"
    assert status["blockers"] == [
        "launch_preparation_immutable_result_conflict"
    ]


def test_worker_refuses_request_bound_to_a_different_deployed_commit(
    tmp_path: Path,
) -> None:
    value, payloads = request_with_fetchable_bytes()
    queue = tmp_path / "queue"
    stage_launch_preparation_request(
        value=value, queue_root=queue, submitted_by="blueprint-webapp"
    )

    def forbidden_fetch(*_args):
        raise AssertionError("commit mismatch must block before object fetch")

    run = process_launch_preparation_queue(
        queue_root=queue,
        input_root=tmp_path / "inputs",
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT,
        source_commit="b" * 40,
        fetcher=forbidden_fetch,
        adapter_materializer=fake_adapter,
    )

    assert run["results"][0]["status"] == "blocked"
    assert run["results"][0]["blockers"] == [
        "launch_preparation_worker_source_commit_mismatch"
    ]
