# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_configuration_output_admission.py
#   src/blueprint_pipeline/task_evaluation_scene_configuration_vast.py
#   src/blueprint_pipeline/control_plane_disk_budget.py
#   src/blueprint_pipeline/control_plane_disk_ledger.py
"""Measured admission for the website scene-configuration provider output (plan 13a.1, 3a1.0).

The ceiling formula refused a paid run unless 5U + 512 MiB of raw space was free,
where U = max(2 x bundle, 1 GB) is the provider's upload ceiling: U for the zip and
4U for a worst-case extraction. The dishwasher scene of 2026-09-27 was refused at
13.14 GB for an archive that, measured on retained runs, is 60-75 MB. Measured
admission holds U + 512 MiB on the disk ledger for the download, makes the zip
durable before extracting it, and sizes the extraction from the zip that came back.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import types
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_disk_budget as disk_budget
from blueprint_pipeline import task_evaluation_artifixer_pretraining as pretraining
from blueprint_pipeline import task_evaluation_scene_configuration_cpu_prestage as cpu_prestage
from blueprint_pipeline import (
    task_evaluation_scene_configuration_output_admission as admission,
)
from blueprint_pipeline import (
    task_evaluation_scene_configuration_provider_artifacts as provider_artifacts,
)
from blueprint_pipeline import task_evaluation_scene_configuration_vast as scene_vast
from blueprint_pipeline import task_evaluation_scene_configuration_warm_bootstrap as warm
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_capacity_recovery import (
    BLOCKER as CEILING_PREALLOCATION_BLOCKER,
    dead_machine_launch_failure,
)
from blueprint_pipeline.task_evaluation_scene_configuration_bundle import (
    BUNDLE_SCHEMA_VERSION,
)
from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import (
    MAX_ATTEMPT_SPEND_USD,
    MAX_HOURLY_RATE_USD,
    MAX_PROVIDER_COMPUTE_SPEND_USD,
    REQUIRED_PARENT_TTL_SECONDS,
)
from tests.test_task_evaluation_launch_preparation_contract import (
    test_configuration_request as _configuration_request,
)
from tests.test_task_evaluation_scene_configuration_bundle import (
    _build,
    _construction_queue,
)
from tests.test_task_evaluation_scene_configuration_cpu_prestage import (
    _fake_entrypoint,
    _prepare,
)

MIB = 1024**2
GIB = 1024**3
RESERVE = 512 * MIB
VOLUME_TOTAL = 100 * GIB
#: The bulk floor on a 100 GiB volume: max(8 GiB, 5 %).
FLOOR = 8 * GIB
#: Upload ceiling of any bundle under 500 MB (the 1 GB minimum).
SMALL_UPLOAD = 1_000_000_000
#: The refused dishwasher run (scene-ec693ebc, 2026-09-27 12:52 UTC).
DISHWASHER_BUNDLE_BYTES = 1_260_479_494
DISHWASHER_FREE_BYTES = 12_519_301_120
DISHWASHER_CEILING_BYTES = 13_141_665_852
CEILING_EXTRACTION_BLOCKER = CEILING_PREALLOCATION_BLOCKER
RESULT_NAME = "task_evaluation_scene_configuration_provider_result.v1.json"
ROLES = {
    "configured_appearance_without_source_object": "appearance.usdc",
    "appearance_removal_receipt": "appearance-receipt.json",
    "appearance_visual_review_receipt": "appearance-review.json",
    "configured_task_thumbnail": "configured-task-thumbnail.png",
    "configured_collision_without_source_object": "collision.usda",
    "collision_excision_receipt": "collision-receipt.json",
    "statically_qualified_replacement_asset": "static.usda",
    "static_qualification_receipt": "static-receipt.json",
    "native_qualified_replacement_asset": "native.usda",
    "native_import_qualification_receipt": "native-receipt.json",
    "configured_scene_bundle_candidate_manifest": "candidate.json",
    "scene_assembly_receipt": "assembly-receipt.json",
}


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


class _Volume:
    """The launch-runs volume as ``shutil.disk_usage`` reports it; tests move ``free``."""

    def __init__(self, free: int, total: int = VOLUME_TOTAL) -> None:
        self.free, self.total = free, total

    def __call__(self, _path):  # type: ignore[no-untyped-def]
        return shutil._ntuple_diskusage(self.total, self.total - self.free, self.free)


def _ledger_rows(ledger: Path) -> list[tuple[str, int]]:
    """(role, expected bytes) of every reservation currently in the ledger."""

    if not ledger.is_dir():
        return []
    rows = [json.loads(path.read_text()) for path in sorted(ledger.glob("*.json"))]
    return sorted((row["role"], row["expected_bytes"]) for row in rows)


def _history(ledger: Path) -> list[dict]:
    path = ledger / "history" / f"{admission.OUTPUT_ROLE}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _thumbnail_review() -> tuple[bytes, bytes]:
    payload = b"exact-reviewed-frame"
    review: dict[str, object] = {
        "schema_version": "task_evaluation_artifixer_ai_visual_review.v1",
        "status": "accepted",
        "review_frame_count": 8,
        "task_thumbnail_is_exact_review_frame": True,
        "task_thumbnail_selection": {
            "camera_id": "camera-3",
            "frame_sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "rationale": "The task surface and configured scene are both clear.",
        },
        "reviewer": {
            "kind": "ai",
            "identity": "artifixer-independent-vision-reviewer-v1",
            "runtime": "openai_agents_sdk",
            "model": "gpt-6-sol",
        },
        "receipt_digest": "",
    }
    review["receipt_digest"] = canonical_digest(review, digest_field="receipt_digest")
    return json.dumps(review, separators=(",", ":")).encode(), payload


def _write_completed_archive(
    path: Path, receipt: dict, *, large_member_bytes: int = 0
) -> None:
    """The provider's completed output: twelve declared roles plus its sealed result."""

    review, thumbnail = _thumbnail_review()
    special = {
        "appearance_visual_review_receipt": review,
        "configured_task_thumbnail": thumbnail,
    }
    rows, members = [], {}
    for role, name in ROLES.items():
        relative = f"stages/stage-1/adapter/{name}"
        payload = special.get(role, (role + "\n").encode())
        members[relative] = payload
        rows.append({
            "role": role,
            "path": "/workspace/runtime_output/" + relative,
            "provider_output_relative_path": relative,
            "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        })
    stages = []
    for index in range(6):
        stage = {
            "schema_version": "task_evaluation_scene_configuration_stage_result.v1",
            "status": "completed",
            "stage_id": f"stage-{index + 1}",
            "canonical_allocator": None,
            "provider_mutations_performed": 0,
            "paid_execution_requested": False,
            "executed_inside_parent_configuration_run": True,
            "raw_secret_values_recorded": False,
            "output_artifacts": rows if index == 0 else [],
            "stage_result_digest": "",
        }
        stage["stage_result_digest"] = canonical_digest(
            stage, digest_field="stage_result_digest"
        )
        stages.append(stage)
    chain = {
        "schema_version": "task_evaluation_scene_configuration_provider_stage_chain.v1",
        "status": "completed",
        "run_id": receipt["run_id"],
        "stage_results": stages,
        "stage_result_digests": [stage["stage_result_digest"] for stage in stages],
        "stage_count": 6,
        "executed_inside_one_parent_provider_run": True,
        "nested_provider_mutations_performed": 0,
        "nested_paid_execution_requested": False,
        "evaluation_episode_executed": False,
        "retry_cap": 0,
        "result_digest": "",
    }
    chain["result_digest"] = canonical_digest(chain, digest_field="result_digest")
    result = {
        "schema_version": "task_evaluation_scene_configuration_provider_result.v1",
        "status": "completed",
        "run_id": receipt["run_id"],
        "source_commit": receipt["source_commit"],
        "source_construction_envelope_digest": receipt["construction_envelope_source_digest"],
        "construction_envelope_digest": receipt["portable_construction_envelope_digest"],
        "stage_chain": chain,
        "evaluation_episode_executed": False,
        "candidate_policy_queried": False,
        "provider_zero_required_after_return": True,
        "blockers": [],
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
        archive.writestr(RESULT_NAME, json.dumps(result))
    if large_member_bytes:
        # A retained ArtiFixer checkpoint: its central directory declares the
        # full size while zeros deflate to a few MB on this disk.
        chunk = bytes(64 * MIB)
        with zipfile.ZipFile(
            path, "a", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as archive:
            with archive.open("ckpt_000.pt", "w", force_zip64=True) as stream:
                remaining = large_member_bytes
                while remaining:
                    written = min(remaining, len(chunk))
                    stream.write(chunk[:written])
                    remaining -= written


def _write_prefix_archive(path: Path, receipt: dict) -> None:
    """What ``cpu_prestage_output.zip`` holds: a sealed completed prefix, not a run."""

    result = {
        "schema_version": "task_evaluation_scene_configuration_provider_result.v1",
        "status": "completed_prefix",
        "run_id": receipt["run_id"],
        "source_commit": receipt["source_commit"],
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(RESULT_NAME, json.dumps(result))
        archive.writestr("stages/stage-1/adapter/appearance.usdc", b"prefix appearance")


def _harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    website: bool = True,
    mode: str | None = admission.MEASURED_MODE,
    bundle_size_bytes: int | None = None,
    free_bytes: int = 90 * GIB,
) -> types.SimpleNamespace:
    """The retry-zero lane with fake staging, provider and object stores.

    Everything the lane decides about disk space is real: the lane's own
    code, the ledger, its history and the extractor.
    """

    tmp_path.mkdir(parents=True, exist_ok=True)
    receipt = _build(tmp_path, "bundle")
    receipt_path = tmp_path / "bundle" / f"{BUNDLE_SCHEMA_VERSION}.receipt.json"
    if bundle_size_bytes is not None:
        real_load = scene_vast.load_scene_configuration_provider_bundle_receipt

        def load(path, **_kwargs):  # type: ignore[no-untyped-def]
            return {**real_load(path), "bundle_size_bytes": bundle_size_bytes}

        monkeypatch.setattr(
            scene_vast, "load_scene_configuration_provider_bundle_receipt", load
        )
    authority_path = tmp_path / "authority.json"
    authority_path.write_text("{}", encoding="utf-8")
    authority = {
        "authority_digest": "sha256:" + "e" * 64,
        "hard_attempt_spend_cap_usd": MAX_ATTEMPT_SPEND_USD,
        "provider_compute_spend_cap_usd": MAX_PROVIDER_COMPUTE_SPEND_USD,
        "maximum_hourly_rate_usd": MAX_HOURLY_RATE_USD,
        "maximum_single_resource_ttl_seconds": REQUIRED_PARENT_TTL_SECONDS,
        "container_image": "nvcr.io/nvidia/isaac-sim@sha256:" + "b" * 64,
        "external_service_spend_caps": {
            "openai": {"maximum_cost_usd": 1.5, "maximum_requests": 32}
        },
    }
    monkeypatch.setattr(
        scene_vast,
        "validate_scene_configuration_paid_authority",
        lambda _value, **_kwargs: authority,
    )
    monkeypatch.setattr(
        scene_vast, "_provider_runtime_inputs", lambda _authority, _receipt=None: ({}, {})
    )
    monkeypatch.setattr(
        scene_vast, "require_paid_resource_admission_grant", lambda *_a, **_k: None
    )
    ledger = tmp_path / "ledger"
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT", str(ledger))
    if mode is None:
        monkeypatch.delenv(admission.OUTPUT_ADMISSION_ENV, raising=False)
    else:
        monkeypatch.setenv(admission.OUTPUT_ADMISSION_ENV, mode)
    volume = _Volume(free_bytes)
    events: list[tuple[str, object]] = []

    def stage(*, job_dir, **kwargs):  # type: ignore[no-untyped-def]
        staging = Path(job_dir)
        staging.mkdir(parents=True, exist_ok=True)
        events.append(("stage", staging.name))
        for name in (
            "provider_bundle_url.txt",
            "provider_output_put_url.txt",
            "provider_output_get_url.txt",
        ):
            (staging / name).write_text(
                f"https://objects.example.test/{name}", encoding="utf-8"
            )
        if staging.name in {"api_pretraining_object_store", "cpu_prestage_object_store"}:
            bundle = Path(kwargs["bundle_path"])
            return {"status": "completed", "provider_bundle_remote_reference": {
                "digest": _sha256(bundle), "size_bytes": bundle.stat().st_size,
                "full_byte_service_account_readback_passed": True}}
        return {"status": "completed"}

    monkeypatch.setattr(scene_vast, "stage_wam_provider_bundle_object_store", stage)
    monkeypatch.setattr(
        scene_vast,
        "arm_independent_vast_watchdog",
        lambda **kwargs: (
            {"status": "armed"},
            types.SimpleNamespace(
                pod_name_prefix=kwargs["pod_name_prefix"] + "fixture-",
                started_instance_id_path=Path(kwargs["job_dir"]) / "started.json",
                deadline_epoch=9_999_999_999.0,
            ),
        ),
    )
    monkeypatch.setattr(
        scene_vast, "close_independent_vast_watchdog",
        lambda **_kwargs: {"status": "provider_terminal"},
    )
    monkeypatch.setattr(
        scene_vast, "cleanup_staged_wam_provider_objects",
        lambda _root: {"all_objects_absent": True},
    )
    monkeypatch.setattr(
        scene_vast, "_consume_authority_once",
        lambda _authority, **_kwargs: {"status": "consumed"},
    )
    monkeypatch.setattr(
        scene_vast, "_stage_owner_only_runtime_secrets", lambda **_kwargs: ({}, None)
    )

    def prepare_semantics(**kwargs):  # type: ignore[no-untyped-def]
        path = Path(kwargs["job_dir"]) / "fixture_pretraining.zip"
        path.write_bytes(b"prepared")
        return {"capsule_path": str(path), "capsule_sha256": _sha256(path),
                "capsule_bytes": path.stat().st_size}

    monkeypatch.setattr(pretraining, "prepare_semantics_before_gpu", prepare_semantics)

    request = _configuration_request()
    request["run_id"] = receipt["run_id"]
    request["expected_production_commit"] = receipt["source_commit"]
    source_envelope = json.loads(
        (tmp_path / "source" / "envelope.json").read_text(encoding="utf-8")
    )
    publication_envelope = {
        "orchestration_id": source_envelope["orchestration_id"],
        "run_id": receipt["run_id"],
        "team_namespace": request["team_namespace"],
        "expected_production_commit": receipt["source_commit"],
        "recipe_digest": source_envelope["recipe_digest"],
        "control_plane_envelope_digest": source_envelope["envelope_digest"],
        "request": request,
        "recipe": {
            "stage_sequence": source_envelope["recipe"]["stage_sequence"],
            "scene_identity": request["scene"]["identity"],
            "task_identity": request["task"]["identity"],
            "subject_identity": request["task"]["subject"]["identity"],
        },
        "render_inputs_result": {
            "status": "derived_method_inputs_materialized",
            "raw_interiorgs_bytes_in_provider_packet": False,
        },
        "provider_disclosure_receipt": {"raw_interiorgs_bytes_in_provider_bundle": False},
    }
    # The lane reads the portable envelope to learn what kind of scene it runs;
    # publication keeps the generic fixture so both modes publish identically.
    lane_envelope = copy.deepcopy(publication_envelope)
    if website:
        lane_envelope["request"]["scene"]["website_native_inputs"] = {
            "prepared_appearance": {"digest": "sha256:" + "1" * 64},
        }
        for row, adapter in zip(
            lane_envelope["recipe"]["stage_sequence"],
            ("website_prepared_appearance", "website_prepared_collision"),
        ):
            row["adapter"] = {"id": adapter, "version": "v1"}
    monkeypatch.setattr(scene_vast, "_portable_construction_envelope", lambda _r: lane_envelope)
    monkeypatch.setattr(
        scene_vast, "_publication_envelope", lambda _receipt, **_kwargs: publication_envelope
    )

    behaviour: dict[str, object] = {
        "output": "completed", "large_member_bytes": 0, "free_after_adapter": None,
    }

    def adapter(**kwargs):  # type: ignore[no-untyped-def]
        provider_run = Path(kwargs["job_dir"])
        provider_run.mkdir(parents=True, exist_ok=True)
        events.append(("adapter", _ledger_rows(ledger)))
        (provider_run / "vast_teardown_manifest.json").write_text(json.dumps(
            {"continuing_spend_from_this_run": False, "vast_instance_ids": [123]}
        ), encoding="utf-8")
        output = Path(kwargs["provider_runtime_output_zip"])
        if behaviour["output"] == "completed":
            _write_completed_archive(
                output, receipt, large_member_bytes=int(behaviour["large_member_bytes"])
            )
        elif behaviour["output"] == "corrupt":
            output.write_bytes(b"not a zip archive")
        if behaviour["free_after_adapter"] is not None:
            volume.free = int(behaviour["free_after_adapter"])
        if behaviour["output"] == "dead":
            value = {"status": "blocked", "blockers": ["vast_heartbeat_instance_exited"],
                     "vast_instance_ids": [123], "provider_create_attempted": True,
                     "continuing_spend_from_this_run": False}
        else:
            value = {"status": "completed", "vast_instance_ids": [123],
                     "provider_create_attempted": True}
        (provider_run / "vast_provider_adapter_result.json").write_text(
            json.dumps(value), encoding="utf-8"
        )
        return value

    monkeypatch.setattr(scene_vast, "run_vast_provider_adapter", adapter)
    object_store = tmp_path / "configured-object-store"
    object_store.mkdir(exist_ok=True)

    def publisher_factory():  # type: ignore[no-untyped-def]
        def publish(*, path: Path, object_name: str):  # type: ignore[no-untyped-def]
            destination = object_store / object_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            return {
                "uri": f"s3://blueprint-inputs/{object_name}",
                "digest": _sha256(path),
                "size_bytes": path.stat().st_size,
                "full_byte_service_account_readback_passed": True,
                "readback_digest": _sha256(destination),
                "readback_size_bytes": destination.stat().st_size,
            }

        return publish

    monkeypatch.setattr(scene_vast, "configured_scene_object_store_publisher", publisher_factory)
    durable: dict[str, object] = {"fail": None}

    def durable_publisher(*, path: Path, artifact_kind: str):  # type: ignore[no-untyped-def]
        # The same B2 CAS publisher also makes the provider bundle durable
        # before allocation; only the provider output is observed here.
        if artifact_kind == "provider-output":
            events.append(("publish", Path(path).name))
            if durable["fail"] is not None:
                raise RuntimeError(str(durable["fail"]))
        digest = _sha256(Path(path))
        destination = object_store / "artifacts" / digest.removeprefix("sha256:")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        return {
            "schema_version": "task_evaluation_scene_artifact_reference.v1",
            "status": "remote_verified",
            "artifact_kind": artifact_kind,
            "uri": f"s3://blueprint-inputs/artifacts/{destination.name}",
            "digest": digest,
            "size_bytes": Path(path).stat().st_size,
            "cache_hit": False,
            "upload_performed": True,
            "content_addressed_key": True,
            "remote_identity_verified": True,
            "full_byte_service_account_readback_passed": True,
            "readback_digest": _sha256(destination),
            "readback_size_bytes": destination.stat().st_size,
            "raw_secret_values_recorded": False,
        }

    monkeypatch.setattr(provider_artifacts, "publish_configured_scene_artifact", durable_publisher)
    original_extract = scene_vast._extract_provider_output

    def extract(archive_path, destination, **kwargs):  # type: ignore[no-untyped-def]
        events.append(("extract", Path(archive_path).name))
        return original_extract(archive_path, destination, **kwargs)

    monkeypatch.setattr(scene_vast, "_extract_provider_output", extract)
    job = tmp_path / "job"

    def run(**overrides):  # type: ignore[no-untyped-def]
        return scene_vast.run_scene_configuration_vast(
            job_dir=job,
            bundle_receipt_path=receipt_path,
            paid_attempt_authority_path=authority_path,
            paid_resource_admission_grant=object(),
            execute=True,
            scene_construction_queue_root=_construction_queue(tmp_path),
            disk_usage_provider=volume,
            **overrides,
        )

    return types.SimpleNamespace(
        receipt=receipt, receipt_path=receipt_path, volume=volume, ledger=ledger,
        events=events, behaviour=behaviour, durable=durable, run=run, job=job,
    )


def _install_prestage(
    lane: types.SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    work: Path,
    *,
    prefix_output: bool = False,
) -> dict:
    """A CPU prefix that reserves its real need on the same ledger and volume."""

    monkeypatch.setenv(cpu_prestage.WORK_DIR_ENV, str(work))
    observed: dict[str, object] = {}
    need = admission.cpu_prefix_peak_bytes(Path(lane.receipt["bundle_path"]))

    def prepare(*, bundle_receipt, job_dir, **_kwargs):  # type: ignore[no-untyped-def]
        observed["ledger_before"] = _ledger_rows(lane.ledger)
        with disk_budget.reserve_control_plane_disk(
            "cpu_prestage", target_root=work, expected_bytes=need,
            reservation_root=lane.ledger, disk_usage=lane.volume, workload="cpu_prestage",
        ):
            observed["ledger_during"] = _ledger_rows(lane.ledger)
            capsule = Path(job_dir) / "cpu_prestage_capsule.zip"
            with zipfile.ZipFile(capsule, "w") as archive:
                archive.writestr("cpu_prestage_transport.json", "{}")
            if prefix_output:
                _write_prefix_archive(Path(job_dir) / "cpu_prestage_output.zip", bundle_receipt)
        return {"capsule_path": str(capsule), "capsule_sha256": _sha256(capsule),
                "capsule_bytes": capsule.stat().st_size}

    monkeypatch.setattr(cpu_prestage, "prestage_stage_limit", lambda _receipt, _env: "stage-2")
    monkeypatch.setattr(cpu_prestage, "prestage_ttl_seconds", lambda _receipt, _limit: 3_600)
    monkeypatch.setattr(cpu_prestage, "prepare_stage_prefix_before_gpu", prepare)
    observed["need"] = need
    return observed


def _event_names(events: list[tuple[str, object]]) -> list[tuple[str, object]]:
    return [event for event in events if event[0] in {"publish", "extract"}]


# ---------------------------------------------------------------------------
# The flag
# ---------------------------------------------------------------------------


def test_scene_configuration_output_role_is_declared_for_a_paid_run() -> None:
    """2 GiB declared; its entry outlives the dispatcher's 5 h start timeout."""

    assert disk_budget.ROLE_FOOTPRINT_BYTES[admission.OUTPUT_ROLE] == 2 * GIB
    assert disk_budget.ROLE_TTL_SECONDS[admission.OUTPUT_ROLE] == 6 * 3600
    assert admission.OUTPUT_ROLE == "scene_configuration_output"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, "ceiling"), ("", "ceiling"), ("ceiling", "ceiling"), ("measured", "measured"),
     ("Measured", None), ("stream", None), (" measured", None)],
)
def test_admission_mode_defaults_to_ceiling_and_refuses_anything_else(raw, expected) -> None:
    environment = {} if raw is None else {admission.OUTPUT_ADMISSION_ENV: raw}
    assert admission.configured_output_admission_mode(environment) == expected


def test_invalid_admission_mode_refuses_before_staging(tmp_path, monkeypatch) -> None:
    lane = _harness(tmp_path, monkeypatch, mode="streamed")

    result = lane.run()

    assert result["status"] == "blocked"
    assert result["blockers"] == ["scene_configuration_output_admission_mode_invalid"]
    assert result["provider_mutations_performed"] == 0
    assert result["continuing_spend_from_this_run"] is False
    assert result["provider_output_disk_capacity"]["schema_version"] == (
        admission.ADMISSION_SCHEMA_VERSION
    )
    assert lane.events == []
    assert not lane.ledger.exists()


# ---------------------------------------------------------------------------
# Pre-allocation hold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["ceiling", "measured"])
def test_dishwasher_admitted_under_measured_mode(tmp_path, monkeypatch, mode) -> None:
    """The observed refusal: 12,519,301,120 free, 13,141,665,852 required.

    ``ceiling`` still refuses it byte for byte. ``measured`` holds U + 512 MiB
    (3,057,829,900 bytes) against the 3,929,366,528 bytes above the floor.
    """

    lane = _harness(
        tmp_path, monkeypatch, mode=mode,
        bundle_size_bytes=DISHWASHER_BUNDLE_BYTES, free_bytes=DISHWASHER_FREE_BYTES,
    )
    upload = 2 * DISHWASHER_BUNDLE_BYTES
    hold = upload + RESERVE
    assert (upload, hold) == (2_520_958_988, 3_057_829_900)
    assert DISHWASHER_FREE_BYTES - FLOOR == 3_929_366_528

    result = lane.run()

    assert result["expected_provider_upload_bytes"] == upload
    if mode == "ceiling":
        assert result["status"] == "blocked"
        assert result["blockers"] == [CEILING_PREALLOCATION_BLOCKER]
        capacity = result["provider_output_disk_capacity"]
        assert capacity["schema_version"] == "scene_configuration_provider_output_disk_capacity.v1"
        assert capacity["required_free_bytes"] == DISHWASHER_CEILING_BYTES == 5 * upload + RESERVE
        assert capacity["observed_free_bytes"] == DISHWASHER_FREE_BYTES
        assert capacity["required_free_bytes"] - capacity["observed_free_bytes"] == 622_364_732
        assert lane.events == []
        assert not lane.ledger.exists()
        return

    assert result["status"] == "completed", result["blockers"]
    assert result["provider_output_admission_mode"] == "measured"
    record = result["provider_output_disk_capacity"]["before_allocation_and_staging"]
    assert record["schema_version"] == admission.ADMISSION_SCHEMA_VERSION
    assert record["status"] == "ready"
    assert record["hold_bytes"] == hold
    assert record["required_available_bytes"] == hold
    assert record["hold"] == "held"
    assert record["hold_reservation"]["expected_bytes"] == hold
    assert record["available_bytes"] == DISHWASHER_FREE_BYTES - FLOOR
    # The hold stood alone on the ledger while the provider ran.
    assert ("adapter", [(admission.OUTPUT_ROLE, hold)]) in lane.events
    # Seventy-odd kilobytes came back: the extraction fit inside the hold.
    extraction = result["provider_output_disk_capacity"]["before_extraction"]
    assert extraction["growth_bytes"] == 0 and extraction["growth_reservation"] is None
    # Released with the sealed result, as a completed sample of the real footprint.
    assert _ledger_rows(lane.ledger) == []
    [sample] = _history(lane.ledger)
    assert sample["outcome"] == "completed"
    assert sample["reserved_bytes"] == hold
    assert 0 < sample["observed_bytes"] < 64 * MIB


@pytest.mark.parametrize("room", ["larger_need", "short"])
def test_prestage_and_output_hold_are_sequential_not_additive(
    tmp_path, monkeypatch, room
) -> None:
    """The CPU prefix reserves 3 x unpacked + 512 MiB on the same volume first.

    Admission checks max(prefix, hold) up front; the hold is taken only after
    the prefix released its own reservation, so the two never add up.
    """

    lane = _harness(tmp_path, monkeypatch)
    work = tmp_path / "prestage-work"
    work.mkdir()
    observed = _install_prestage(lane, monkeypatch, work)
    hold = SMALL_UPLOAD + RESERVE
    need = int(observed["need"])
    larger, smaller = max(hold, need), min(hold, need)
    if room == "larger_need":
        lane.volume.free = FLOOR + larger + smaller // 2
        # Together they would not fit: a hold held across the prefix would
        # have refused the prefix's own reservation.
        assert lane.volume.free - FLOOR < hold + need
    else:
        lane.volume.free = FLOOR + larger - 1

    result = lane.run(cpu_prestage_stage_limit="stage-2")

    if room == "short":
        assert result["status"] == "blocked"
        assert result["blockers"] == [admission.BUDGET_EXCEEDED_BLOCKER]
        assert result["provider_mutations_performed"] == 0
        record = result["provider_output_disk_capacity"]
        assert record["required_available_bytes"] == larger
        assert record["available_bytes"] == larger - 1
        assert lane.events == [] and "ledger_before" not in observed
        # The refusal carries exactly what capacity recovery re-opens.
        assert admission.recorded_preallocation_refusal(
            result, maximum_archive_bytes=SMALL_UPLOAD, job_dir=lane.job.resolve()
        ) == record
        return
    assert result["status"] == "completed", result["blockers"]
    assert observed["ledger_before"] == []
    assert observed["ledger_during"] == [("cpu_prestage", need)]
    assert ("adapter", [(admission.OUTPUT_ROLE, hold)]) in lane.events
    record = result["provider_output_disk_capacity"]["before_allocation_and_staging"]
    assert record["required_available_bytes"] == larger
    assert record["hold"] == "held"
    assert record["hold_phase"] == "after_cpu_prefix"
    assert record["sequential_phases"] == [
        {"phase": "cpu_prestage", "peak_bytes": need, "shares_output_volume": True}
    ]
    assert _ledger_rows(lane.ledger) == []


def test_up_front_prefix_need_is_what_the_prestage_reserves(tmp_path) -> None:
    receipt, _job = _prepare(tmp_path, _fake_entrypoint(["stage-1", "stage-2", "stage-3", "stage-4"]))

    assert admission.cpu_prefix_peak_bytes(tmp_path / "bundle.zip") == receipt["reserved_peak_bytes"]


# ---------------------------------------------------------------------------
# Durable before extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "outcome", ["downloaded", "cpu_prefix", "corrupt", "extraction_refused"]
)
def test_archive_is_published_before_extraction_on_every_outcome(
    tmp_path, monkeypatch, outcome
) -> None:
    """The local zip, from whichever source, reaches B2 before anything is extracted.

    A downloaded and an SSH-recovered zip land at the same path, so the
    downloaded case covers both.
    """

    lane = _harness(tmp_path, monkeypatch)
    expected_name = "vast_provider_runtime_output.zip"
    if outcome == "cpu_prefix":
        work = tmp_path / "prestage-work"
        work.mkdir()
        _install_prestage(lane, monkeypatch, work, prefix_output=True)
        lane.behaviour["output"] = "dead"
        expected_name = "cpu_prestage_output.zip"
    elif outcome == "corrupt":
        lane.behaviour["output"] = "corrupt"
    elif outcome == "extraction_refused":
        lane.behaviour["large_member_bytes"] = 1_100_000_000
        lane.behaviour["free_after_adapter"] = FLOOR + SMALL_UPLOAD + RESERVE

    result = lane.run(
        **({"cpu_prestage_stage_limit": "stage-2"} if outcome == "cpu_prefix" else {})
    )

    order = _event_names(lane.events)
    assert order[0] == ("publish", expected_name)
    if outcome == "extraction_refused":
        assert order == [("publish", expected_name)]
    else:
        assert order == [("publish", expected_name), ("extract", expected_name)]
    assert result["provider_output_archive_durable"] is True
    reference = result["provider_runtime_output_remote_reference"]
    assert reference["remote_identity_verified"] is True
    assert Path(result["provider_runtime_output_remote_index_path"]).is_file()
    assert result["status"] == ("completed" if outcome == "downloaded" else "blocked")


def test_a_publish_failure_keeps_todays_blocker_and_records_the_archive_not_durable(
    tmp_path, monkeypatch
) -> None:
    lane = _harness(tmp_path, monkeypatch)
    lane.durable["fail"] = "fixture object store unavailable"

    result = lane.run()

    assert result["status"] == "blocked"
    assert result["provider_output_archive_durable"] is False
    assert any(
        blocker.startswith("scene_configuration_provider_output_durable_publication_failed:")
        for blocker in result["blockers"]
    )
    # The extraction still ran from the local zip, which stays for recovery.
    assert _event_names(lane.events) == [
        ("publish", "vast_provider_runtime_output.zip"),
        ("extract", "vast_provider_runtime_output.zip"),
    ]
    assert Path(result["provider_runtime_output_zip_path"]).is_file()
    assert _ledger_rows(lane.ledger) == []


# ---------------------------------------------------------------------------
# Extraction sized from the actual zip
# ---------------------------------------------------------------------------


def test_extraction_is_sized_from_the_actual_zip(tmp_path) -> None:
    """A 70 MB archive needs about 0.6 GB to extract, not 4U + 512 MiB."""

    archive = tmp_path / "vast_provider_runtime_output.zip"
    payload = bytes(70_000_000)
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("stages/stage-1/adapter/appearance.usdc", payload)
        zipped.writestr(RESULT_NAME, b"{}")
    assert 70_000_000 < archive.stat().st_size < 70_001_000

    requirement = admission.extraction_requirement(
        archive, maximum_archive_bytes=SMALL_UPLOAD
    )

    assert requirement["member_count"] == 2
    assert requirement["expanded_bytes"] == 70_000_002
    assert requirement["extraction_bytes"] == 70_000_002
    assert requirement["required_bytes"] == 70_000_002 + RESERVE
    assert 0.60e9 < requirement["required_bytes"] < 0.61e9
    ceiling = provider_artifacts._provider_output_disk_requirements(SMALL_UPLOAD)
    assert ceiling["required_free_bytes_before_extraction"] == 4 * SMALL_UPLOAD + RESERVE


@pytest.mark.parametrize("fault", ["absent", "unreadable", "over_ceiling"])
def test_extraction_needs_no_member_bytes_when_the_extractor_refuses_first(
    tmp_path, fault
) -> None:
    """Only a zip the extractor will unpack is sized; the others keep its typed refusal."""

    archive = tmp_path / "vast_provider_runtime_output.zip"
    if fault == "unreadable":
        archive.write_bytes(b"not a zip archive")
    elif fault == "over_ceiling":
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("evidence.txt", bytes(4096))

    requirement = admission.extraction_requirement(
        archive, maximum_archive_bytes=1024 if fault == "over_ceiling" else SMALL_UPLOAD
    )

    assert requirement["extraction_bytes"] == 0
    assert requirement["required_bytes"] == RESERVE


def test_extraction_refusal_blocks_with_the_archive_durable(tmp_path, monkeypatch) -> None:
    """An archive that outgrows the hold takes a growth reservation or blocks.

    The zip is already in B2 and stays on this host; nothing is extracted.
    """

    lane = _harness(tmp_path, monkeypatch)
    lane.behaviour["large_member_bytes"] = 1_100_000_000
    # After the download the volume has no room left beyond the hold.
    lane.behaviour["free_after_adapter"] = FLOOR + SMALL_UPLOAD + RESERVE

    result = lane.run()

    assert result["status"] == "blocked"
    assert admission.EXTRACTION_BUDGET_EXCEEDED_BLOCKER in result["blockers"]
    assert result["provider_output_archive_durable"] is True
    assert result["provider_runtime_output_remote_reference"]["digest"] == (
        result["provider_runtime_output_zip_sha256"]
    )
    assert Path(result["provider_runtime_output_zip_path"]).is_file()
    assert not (lane.job / "immutable_execution").exists()
    extraction = result["provider_output_disk_capacity"]["before_extraction"]
    assert extraction["schema_version"] == admission.EXTRACTION_SCHEMA_VERSION
    assert extraction["status"] == "blocked"
    assert extraction["blockers"] == [admission.EXTRACTION_BUDGET_EXCEEDED_BLOCKER]
    assert extraction["archive_durable"] is True
    assert extraction["expanded_bytes"] > 1_100_000_000
    assert extraction["growth_bytes"] > 0
    assert extraction["hold_bytes_remaining"] < extraction["required_bytes"]
    assert result["configured_scene_published"] is False
    assert _ledger_rows(lane.ledger) == []
    assert _history(lane.ledger)[-1]["outcome"] == "blocked"


def test_an_archive_that_outgrows_the_hold_extracts_under_a_growth_reservation(
    tmp_path, monkeypatch
) -> None:
    lane = _harness(tmp_path, monkeypatch)
    lane.behaviour["large_member_bytes"] = 1_100_000_000
    growth_seen: list[list[tuple[str, int]]] = []

    def extract(archive_path, destination, **kwargs):  # type: ignore[no-untyped-def]
        growth_seen.append(_ledger_rows(lane.ledger))
        # Stop before writing a gigabyte of zeros to this disk.
        return {}, ["fixture_extraction_stopped"]

    monkeypatch.setattr(scene_vast, "_extract_provider_output", extract)

    result = lane.run()

    extraction = result["provider_output_disk_capacity"]["before_extraction"]
    assert extraction["status"] == "ready"
    growth = extraction["growth_bytes"]
    assert growth > 0
    assert extraction["growth_reservation"]["expected_bytes"] == growth
    hold = SMALL_UPLOAD + RESERVE
    assert growth_seen == [sorted([(admission.OUTPUT_ROLE, hold), (admission.OUTPUT_ROLE, growth)])]
    assert extraction["hold_bytes_remaining"] + growth == extraction["required_bytes"]
    assert _ledger_rows(lane.ledger) == []


# ---------------------------------------------------------------------------
# Which runs use it
# ---------------------------------------------------------------------------


def _legacy_passthrough_record(result: dict) -> None:
    capacity = result["provider_output_disk_capacity"]
    records = capacity if "schema_version" not in capacity else {"refusal": capacity}
    for record in records.values():
        assert record["schema_version"] == "scene_configuration_provider_output_disk_capacity.v1"
    assert "provider_output_admission_mode" not in result
    assert "provider_output_archive_durable" not in result


@pytest.mark.parametrize("mode", [None, "", "ceiling"])
def test_ceiling_mode_is_byte_identical(tmp_path, monkeypatch, mode) -> None:
    """Unset, empty and ``ceiling`` run today's path: legacy formula, extract then
    publish, no ledger. The lane tests at test_task_evaluation_scene_configuration_bundle
    :3248, :3524, :3906 and :4670 pass unedited in this mode."""

    lane = _harness(tmp_path, monkeypatch, mode=mode)

    result = lane.run()

    assert result["status"] == "completed", result["blockers"]
    _legacy_passthrough_record(result)
    capacity = result["provider_output_disk_capacity"]
    assert capacity["before_allocation_and_staging"]["required_free_bytes"] == (
        5 * SMALL_UPLOAD + RESERVE
    )
    assert capacity["before_extraction"]["required_free_bytes"] == 4 * SMALL_UPLOAD + RESERVE
    assert _event_names(lane.events) == [
        ("extract", "vast_provider_runtime_output.zip"),
        ("publish", "vast_provider_runtime_output.zip"),
    ]
    assert not lane.ledger.exists()

    gate = admission.open_scene_configuration_output_admission(
        job=tmp_path / "unit-job", receipt=lane.receipt,
        read_envelope=lambda _r: pytest.fail("ceiling mode must not read the envelope"),
        expected_upload_bytes=SMALL_UPLOAD, diagnostic_only=False, retain_warm_session=False,
        api_pretraining=False, cpu_prestage=False,
        environment={} if mode is None else {admission.OUTPUT_ADMISSION_ENV: mode},
    )
    sentinel = object()
    assert gate.before_allocation(lambda **kwargs: (sentinel, kwargs), a=1) == (sentinel, {"a": 1})
    assert gate.hold_before_allocation() is None
    arguments = {"maximum_archive_bytes": SMALL_UPLOAD, "extractor": sentinel,
                 "diagnostic_only": False, "disk_usage_provider": None}
    assert gate.extract(
        lambda *args, **kwargs: (sentinel, args, kwargs), Path("a.zip"), Path("out"), **arguments
    ) == (sentinel, (Path("a.zip"), Path("out")), arguments)
    assert gate.publishes({}) is False and gate.publishes({"status": "blocked"}) is True
    assert gate.result_fields() == {}
    sealed = {"status": "completed"}
    assert admission.release_scene_configuration_output(tmp_path / "unit-job", sealed) is sealed


@pytest.mark.parametrize("run_kind", ["non_website", "diagnostic", "warm_session", "website"])
def test_non_website_and_diagnostic_runs_stay_on_ceiling(
    tmp_path, monkeypatch, run_kind
) -> None:
    """With ``measured`` set, only a production website run leaves the ceiling formula.

    At the dishwasher's free space the ceiling refuses before staging and
    measured admits, so the path each run took is visible in its result.
    """

    lane = _harness(
        tmp_path, monkeypatch, website=run_kind != "non_website",
        bundle_size_bytes=DISHWASHER_BUNDLE_BYTES, free_bytes=DISHWASHER_FREE_BYTES,
    )
    overrides: dict[str, object] = {}
    if run_kind == "diagnostic":
        real_load = scene_vast.load_scene_configuration_provider_bundle_receipt
        monkeypatch.setattr(
            scene_vast, "load_scene_configuration_provider_bundle_receipt",
            lambda path, **_k: {**real_load(path), "bundle_size_bytes": DISHWASHER_BUNDLE_BYTES},
        )
        monkeypatch.setattr(
            scene_vast, "diagnostic_parent_runtime_budget_blockers", lambda **_k: []
        )
        overrides["diagnostic_only"] = True
    elif run_kind == "warm_session":
        monkeypatch.setattr(warm, "validate_warm_bootstrap_request", lambda **_k: {})
        overrides.update(
            retain_warm_session=True,
            warm_session_authority_path=tmp_path / "warm-authority.json",
            warm_session_output_root=tmp_path / "warm-output",
        )

    result = lane.run(**overrides)

    if run_kind == "website":
        assert result["status"] == "completed", result["blockers"]
        assert result["provider_output_admission_mode"] == "measured"
        return
    assert result["blockers"] == [CEILING_PREALLOCATION_BLOCKER]
    assert result["provider_mutations_performed"] == 0
    _legacy_passthrough_record(result)
    assert result["provider_output_disk_capacity"]["required_free_bytes"] == DISHWASHER_CEILING_BYTES
    assert lane.events == []
    assert not lane.ledger.exists()


@pytest.mark.parametrize(
    ("diagnostic_only", "retain_warm_session", "website", "expected"),
    [(False, False, True, True), (True, False, True, False), (False, True, True, False),
     (False, False, False, False)],
)
def test_only_production_website_runs_are_eligible(
    tmp_path, diagnostic_only, retain_warm_session, website, expected
) -> None:
    envelope = {"request": {"scene": {"website_native_inputs": {"x": 1} if website else None}}}
    gate = admission.open_scene_configuration_output_admission(
        job=tmp_path / "job", receipt={"bundle_path": str(tmp_path / "missing.zip")},
        read_envelope=lambda _r: envelope, expected_upload_bytes=SMALL_UPLOAD,
        diagnostic_only=diagnostic_only, retain_warm_session=retain_warm_session,
        api_pretraining=False, cpu_prestage=False,
        environment={admission.OUTPUT_ADMISSION_ENV: "measured"},
    )
    assert gate.measured is expected


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def test_dead_machine_retry_accepts_new_codes(tmp_path, monkeypatch) -> None:
    """A machine that died produced nothing; measured mode must not change what
    that looks like to the auto-retry, and its new codes classify like the
    ceiling codes they replace."""

    results = {}
    for mode in ("ceiling", "measured"):
        lane = _harness(tmp_path / mode, monkeypatch, mode=mode)
        lane.behaviour["output"] = "dead"
        results[mode] = lane.run()
    measured, ceiling = results["measured"], results["ceiling"]

    assert measured["status"] == ceiling["status"] == "blocked"
    assert measured["blockers"] == ceiling["blockers"]
    assert "vast_heartbeat_instance_exited" in measured["blockers"]
    assert dead_machine_launch_failure(measured) is True
    assert dead_machine_launch_failure(ceiling) is True
    assert measured["provider_output_archive_durable"] is False
    # Each measured refusal code is treated exactly like the ceiling code it replaces.
    proof = ["vast_heartbeat_instance_exited", "scene_configuration_provider_not_completed"]
    for new, old in (
        (admission.BUDGET_EXCEEDED_BLOCKER, CEILING_PREALLOCATION_BLOCKER),
        (admission.EXTRACTION_BUDGET_EXCEEDED_BLOCKER, CEILING_EXTRACTION_BLOCKER),
        (admission.ADMISSION_UNAVAILABLE_BLOCKER,
         "scene_configuration_provider_output_disk_capacity_unavailable"),
    ):
        for blockers in ([new], [*proof, new]):
            twin = [old if blocker == new else blocker for blocker in blockers]
            assert dead_machine_launch_failure({**measured, "blockers": blockers}) is (
                dead_machine_launch_failure({**measured, "blockers": twin})
            )
