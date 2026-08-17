from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.fresh_scene_semantic_teacher_image_edit import (
    PACKET_SCHEMA_VERSION,
)
from blueprint_pipeline.semantic_teacher_image_edit_bundle import (
    BUNDLE_RECEIPT_SCHEMA_VERSION,
    ENTRYPOINT,
    MANIFEST_SCHEMA_VERSION,
    SemanticTeacherImageEditBundleError,
    build_semantic_teacher_image_edit_provider_bundle,
)
from blueprint_pipeline.semantic_teacher_image_edit_paid_authority import (
    consume_semantic_teacher_image_edit_paid_authority_once,
    materialize_semantic_teacher_image_edit_paid_authority,
    validate_semantic_teacher_image_edit_paid_authority,
)
from blueprint_pipeline.semantic_teacher_image_edit_paid_lane import (
    materialize_semantic_teacher_image_edit_result,
    materialize_semantic_teacher_no_allocation_closeout,
    materialize_semantic_teacher_provider_zero_receipt,
    prepare_semantic_teacher_image_edit_allocator_dry_run,
)
from blueprint_pipeline.semantic_teacher_image_edit_worker import (
    RUNTIME_RESULT_SCHEMA_VERSION,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, root: Path | None = None) -> dict:
    result = {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    result["relative_path" if root is not None else "path"] = (
        path.relative_to(root).as_posix() if root is not None else str(path)
    )
    return result


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _packet(
    tmp_path: Path,
    *,
    task_count: int,
    cameras_per_task: int = 2,
    maximum_cost_per_request_usd: float = 0.1,
) -> Path:
    root = tmp_path / f"packet-{task_count}"
    root.mkdir(parents=True)
    registry_entry = {
        "backend_id": "fixture_hosted_semantic_editor",
        "capability": "semantic_teacher_image_edit",
        "model_identity": "fixture-immutable-model-revision",
        "license": "fixture-commercial",
        "license_url": "https://license.example.invalid/fixture",
        "commercial_use_permitted": True,
        "recorded_on": "2026-08-13",
        "execution": {
            "adapter_id": "openai_images_edits_v1",
            "provider_id": "fixture-provider",
            "transport_kind": "hosted_image_edit",
            "endpoint": "https://editor.example.invalid/v1/images/edits",
            "model_snapshot": "fixture-immutable-model-revision",
            "runtime_image_identity": "registry.example.invalid/semantic-worker@sha256:"
            + "a" * 64,
            "masked_image_edit_supported": True,
            "high_fidelity_input_supported": True,
            "input_fidelity_parameter_supported": True,
            "output_formats": ["png"],
            "mask_encoding": "rgba_alpha_zero_edit_region_png",
            "external_disclosure_required": True,
            "supported_output_sizes": ["8x6"],
            "pricing_binding": {
                "kind": "immutable_registry_rate",
                "billing_unit": "per_request",
                "currency": "USD",
                "pricing_identity": "fixture-price-2026-08-13",
                "max_cost_per_request_usd": maximum_cost_per_request_usd,
                "usage_required": False,
                "usd_per_million_tokens": {},
            },
            "default_options": {"quality": "high", "output_format": "png"},
        },
    }
    backend_digest = canonical_digest(registry_entry)
    registry = root / "image_editor_backend_registry.json"
    _write_json(
        registry,
        {
            "schema_version": "image_editor_backend_registry.v1",
            "backends": [registry_entry],
        },
    )
    rights_value = {
        "schema_version": "fresh_scene_semantic_teacher_image_edit_rights.v1",
        "status": "accepted_for_private_derived_semantic_edit",
        "accepted_by": "fixture-human",
        "accepted_on": "2026-08-13",
        "source_candidate_inputs_receipt_digest": "sha256:" + "1" * 64,
        "publisher_scene_id": "840920",
        "backend_id": registry_entry["backend_id"],
        "backend_entry_digest": backend_digest,
        "provider_id": "fixture-provider",
        "model_snapshot": "fixture-immutable-model-revision",
        "raw_nonredistributable_source_bytes_included": False,
        "issued_by_agent": False,
        "private_derived_frame_disclosure_authorized": True,
        "provider_retention_terms_accepted": True,
        "provider_training_terms_accepted": True,
        "attestation_digest": "",
    }
    rights_value["attestation_digest"] = canonical_digest(
        rights_value, digest_field="attestation_digest"
    )
    rights = root / "human_rights_attestation.json"
    _write_json(rights, rights_value)
    tasks = []
    for task_index in range(task_count):
        task_id = f"task_{task_index}"
        frames = []
        for frame_index in range(cameras_per_task):
            rgb = root / "tasks" / task_id / "input_frames" / f"{frame_index:05d}.png"
            mask = root / "tasks" / task_id / "edit_masks" / f"{frame_index:05d}.png"
            rgb.parent.mkdir(parents=True, exist_ok=True)
            mask.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB",
                (8, 6),
                (10 + task_index, 20 + frame_index, 30),
            ).save(rgb)
            Image.new("RGBA", (8, 6), (255, 255, 255, 0)).save(mask)
            frames.append(
                {
                    "frame_index": frame_index,
                    "camera_id": f"camera_{frame_index}",
                    "width": 8,
                    "height": 6,
                    "staged_input_rgb": _record(rgb, root=root),
                    "staged_edit_mask": _record(mask, root=root),
                }
            )
        tasks.append(
            {
                "task_id": task_id,
                "camera_count": cameras_per_task,
                "frames": frames,
            }
        )
    packet = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "status": "semantic_teacher_image_edit_packet_prepared_no_upload_no_execution",
        "backend_registry": {
            **_record(registry),
            "selected_backend_entry_digest": backend_digest,
        },
        "rights_attestation": {
            **_record(rights),
            "attestation_digest": rights_value["attestation_digest"],
        },
        "backend": {
            "registry_entry": registry_entry,
            "backend_entry_digest": backend_digest,
            "execution": registry_entry["execution"],
            "prompt_policy": "fixture-policy-v1",
            "prompt": "Remove the selected object and reconstruct the empty room.",
        },
        "task_count": task_count,
        "request_count": task_count * cameras_per_task,
        "retry_count": 0,
        "tasks": tasks,
        "provider_mutations_performed": 0,
        "raw_nonredistributable_source_bytes_included": False,
        "private_derived_calibrated_frames_included": True,
        "packet_digest": "",
    }
    packet["packet_digest"] = canonical_digest(packet, digest_field="packet_digest")
    packet_path = root / f"{PACKET_SCHEMA_VERSION}.json"
    _write_json(packet_path, packet)
    return packet_path


def _committed_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    worker = repository / "src/blueprint_pipeline/semantic_teacher_image_edit_worker.py"
    worker.parent.mkdir(parents=True)
    shutil.copyfile(
        REPOSITORY_ROOT / "src/blueprint_pipeline/semantic_teacher_image_edit_worker.py",
        worker,
    )
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Fixture"], cwd=repository, check=True
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "fixture worker"], cwd=repository, check=True
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repository, commit


def _build(
    tmp_path: Path,
    *,
    task_count: int = 2,
    cameras_per_task: int = 2,
    maximum_cost_per_request_usd: float = 0.1,
    suffix: str = "a",
) -> tuple[dict, Path, Path]:
    packet = _packet(
        tmp_path / suffix,
        task_count=task_count,
        cameras_per_task=cameras_per_task,
        maximum_cost_per_request_usd=maximum_cost_per_request_usd,
    )
    repository, commit = _committed_repository(tmp_path / f"repo-{suffix}")
    output = tmp_path / f"bundle-{suffix}"
    receipt = build_semantic_teacher_image_edit_provider_bundle(
        packet_path=packet,
        repository_root=repository,
        expected_source_commit=commit,
        output_root=output,
    )
    bundle = Path(receipt["bundle"]["path"])
    receipt_path = output / f"{BUNDLE_RECEIPT_SCHEMA_VERSION}.json"
    return receipt, bundle, receipt_path


@pytest.mark.parametrize("task_count", [1, 2, 5])
def test_bundle_supports_one_to_five_tasks_and_all_cameras(
    tmp_path: Path, task_count: int
) -> None:
    receipt, bundle, _receipt_path = _build(
        tmp_path, task_count=task_count, suffix=f"count-{task_count}"
    )
    assert receipt["task_count"] == task_count
    assert receipt["camera_count"] == task_count * 2
    assert receipt["maximum_parallel_requests"] == 4
    assert receipt["rehearsal"]["token_lookup_performed"] is False
    assert receipt["rehearsal"]["upload_performed"] is False
    assert receipt["provider_mutations_performed"] == 0
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        assert ENTRYPOINT in names
        manifest_name = "provider_runtime/semantic_teacher_image_edit_provider_manifest.v1.json"
        manifest = json.loads(archive.read(manifest_name))
        runtime_request = json.loads(
            archive.read(
                "provider_runtime/semantic_teacher_image_edit_runtime_request.v1.json"
            )
        )
        assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
        assert manifest["classification"] == "private_derived_semantic_teacher_image_edit"
        assert manifest["environment"]["secret_values_stored"] is False
        assert manifest["maximum_parallel_requests"] == 4
        assert runtime_request["max_parallel_requests"] == 4
        assert "provider_zero_receipt.json" in manifest["output_allowlist"]
        assert len([name for name in names if name.endswith("/input_frames/00000.png")]) == task_count
        assert not any("publisher" in name.lower() or "token" in name.lower() for name in names)


@pytest.mark.parametrize("max_parallel_requests", [True, 0, 5])
def test_bundle_rejects_invalid_parallel_request_bound(
    tmp_path: Path, max_parallel_requests: object
) -> None:
    packet_path = _packet(tmp_path / "packet", task_count=1)
    repository, commit = _committed_repository(tmp_path / "repo")
    with pytest.raises(
        SemanticTeacherImageEditBundleError,
        match="semantic_teacher_bundle_packet_invalid",
    ):
        build_semantic_teacher_image_edit_provider_bundle(
            packet_path=packet_path,
            repository_root=repository,
            expected_source_commit=commit,
            output_root=tmp_path / "bundle",
            max_parallel_requests=max_parallel_requests,
        )


def test_bundle_is_byte_deterministic(tmp_path: Path) -> None:
    packet = _packet(tmp_path / "shared", task_count=2)
    repository, commit = _committed_repository(tmp_path / "repo")
    bundles = []
    for name in ("first", "second"):
        receipt = build_semantic_teacher_image_edit_provider_bundle(
            packet_path=packet,
            repository_root=repository,
            expected_source_commit=commit,
            output_root=tmp_path / name,
        )
        bundles.append(Path(receipt["bundle"]["path"]).read_bytes())
    assert bundles[0] == bundles[1]


def test_bundle_rejects_changed_packet_input(tmp_path: Path) -> None:
    packet_path = _packet(tmp_path, task_count=1)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    input_path = packet_path.parent / packet["tasks"][0]["frames"][0][
        "staged_input_rgb"
    ]["relative_path"]
    input_path.write_bytes(b"changed")
    repository, commit = _committed_repository(tmp_path / "repo")
    with pytest.raises(
        SemanticTeacherImageEditBundleError,
        match="semantic_teacher_bundle_source_frame_unbound",
    ):
        build_semantic_teacher_image_edit_provider_bundle(
            packet_path=packet_path,
            repository_root=repository,
            expected_source_commit=commit,
            output_root=tmp_path / "bundle",
        )


def test_bundle_requires_clean_tracked_committed_worker(tmp_path: Path) -> None:
    packet_path = _packet(tmp_path / "packet", task_count=1)
    repository, commit = _committed_repository(tmp_path / "repo")
    worker = repository / "src/blueprint_pipeline/semantic_teacher_image_edit_worker.py"
    worker.write_text(worker.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with pytest.raises(
        SemanticTeacherImageEditBundleError,
        match="semantic_teacher_bundle_repository_not_clean_committed",
    ):
        build_semantic_teacher_image_edit_provider_bundle(
            packet_path=packet_path,
            repository_root=repository,
            expected_source_commit=commit,
            output_root=tmp_path / "bundle",
        )


def _authority(tmp_path: Path) -> tuple[dict, Path, Path, dict, Path]:
    receipt, bundle, receipt_path = _build(tmp_path, task_count=2)
    authority_path = tmp_path / "authority.json"
    authority = materialize_semantic_teacher_image_edit_paid_authority(
        bundle_path=bundle,
        bundle_receipt_path=receipt_path,
        authorization_reference="User-directed bounded semantic teacher run",
        authorized_by="fixture-user",
        authorized_on="2026-08-13",
        source_commit_sha=receipt["source_commit_sha"],
        backend_entry_digest=receipt["backend_entry_digest"],
        task_count=2,
        camera_count=4,
        maximum_hourly_rate_usd=0.5,
        hard_total_spend_cap_usd=1.0,
        hard_ttl_seconds=600,
        aggregate_goal_spend_before_attempt_usd=0.0,
        aggregate_goal_spend_cap_usd=20.0,
        output_path=authority_path,
    )
    return authority, authority_path, bundle, receipt, receipt_path


def test_authority_binds_bundle_backend_cardinality_and_caps(tmp_path: Path) -> None:
    authority, _path, bundle, receipt, _receipt_path = _authority(tmp_path)
    validated = validate_semantic_teacher_image_edit_paid_authority(
        authority,
        bundle_path=bundle,
        bundle_receipt=receipt,
        source_commit_sha=authority["source_commit_sha"],
        backend_entry_digest=receipt["backend_entry_digest"],
        task_count=2,
        camera_count=4,
        maximum_hourly_rate_usd=0.5,
        hard_total_spend_cap_usd=1.0,
        hard_ttl_seconds=600,
    )
    assert validated["maximum_provider_allocations"] == 1
    assert validated["maximum_automatic_retries"] == 0
    assert validated["automatic_paid_retry_authorized"] is False
    assert validated["consumption_root_kind"] == "host_private_atomic_single_use"
    assert validated["hosted_editor_spend_upper_bound_usd"] == pytest.approx(
        receipt["camera_count"] * receipt["maximum_cost_per_request_usd"]
    )


def test_authority_admits_ten_dollar_hour_long_sixteen_frame_attempt(
    tmp_path: Path,
) -> None:
    receipt, bundle, receipt_path = _build(
        tmp_path,
        task_count=2,
        cameras_per_task=8,
        maximum_cost_per_request_usd=0.3,
    )
    authority = materialize_semantic_teacher_image_edit_paid_authority(
        bundle_path=bundle,
        bundle_receipt_path=receipt_path,
        authorization_reference="User authorized a larger bounded semantic attempt",
        authorized_by="fixture-user",
        authorized_on="2026-08-16",
        source_commit_sha=receipt["source_commit_sha"],
        backend_entry_digest=receipt["backend_entry_digest"],
        task_count=2,
        camera_count=16,
        maximum_hourly_rate_usd=0.4,
        hard_total_spend_cap_usd=10.0,
        hard_ttl_seconds=3_600,
        aggregate_goal_spend_before_attempt_usd=0.0,
        aggregate_goal_spend_cap_usd=25.0,
        output_path=tmp_path / "authority-cap-ten.json",
    )
    assert authority["hard_total_spend_cap_usd"] == 10.0
    assert authority["hard_ttl_seconds"] == 3_600
    assert authority["hosted_editor_spend_upper_bound_usd"] == pytest.approx(4.8)
    validation_kwargs = {
        "bundle_path": bundle,
        "bundle_receipt": receipt,
        "source_commit_sha": authority["source_commit_sha"],
        "backend_entry_digest": receipt["backend_entry_digest"],
        "task_count": 2,
        "camera_count": 16,
        "maximum_hourly_rate_usd": 0.4,
        "hard_total_spend_cap_usd": 10.0,
        "hard_ttl_seconds": 3_600,
    }
    validate_semantic_teacher_image_edit_paid_authority(
        authority, **validation_kwargs
    )

    hostile_cases = (
        {
            "hard_total_spend_cap_usd": 10.01,
        },
        {
            "hard_ttl_seconds": 3_601,
            "maximum_compute_cost_usd": 0.4 * 3_601 / 3_600,
            "vast_spend_upper_bound_usd": 0.4 * 3_601 / 3_600,
        },
        {
            "maximum_hourly_rate_usd": 10.01,
            "maximum_compute_cost_usd": 10.01,
            "vast_spend_upper_bound_usd": 10.01,
        },
    )
    for changes in hostile_cases:
        forged = {**authority, **changes}
        forged["authorization_digest"] = canonical_digest(
            forged, digest_field="authorization_digest"
        )
        forged_validation_kwargs = {
            **validation_kwargs,
            **{
                key: value
                for key, value in changes.items()
                if key
                in {
                    "maximum_hourly_rate_usd",
                    "hard_total_spend_cap_usd",
                    "hard_ttl_seconds",
                }
            },
        }
        with pytest.raises(
            ValueError, match="semantic_teacher_paid_authority_invalid"
        ):
            validate_semantic_teacher_image_edit_paid_authority(
                forged, **forged_validation_kwargs
            )

    with pytest.raises(
        ValueError, match="semantic_teacher_paid_authority_configuration_invalid"
    ):
        materialize_semantic_teacher_image_edit_paid_authority(
            bundle_path=bundle,
            bundle_receipt_path=receipt_path,
            authorization_reference="Attempt exceeds the authorized lane ceiling",
            authorized_by="fixture-user",
            authorized_on="2026-08-16",
            source_commit_sha=receipt["source_commit_sha"],
            backend_entry_digest=receipt["backend_entry_digest"],
            task_count=2,
            camera_count=16,
            maximum_hourly_rate_usd=0.4,
            hard_total_spend_cap_usd=10.01,
            hard_ttl_seconds=3_600,
            aggregate_goal_spend_before_attempt_usd=0.0,
            aggregate_goal_spend_cap_usd=25.0,
            output_path=tmp_path / "authority-over-cap.json",
        )


def test_authority_rejects_nonzero_unreconciled_prior_spend(tmp_path: Path) -> None:
    receipt, bundle, receipt_path = _build(tmp_path, task_count=1)
    with pytest.raises(
        ValueError, match="semantic_teacher_paid_authority_configuration_invalid"
    ):
        materialize_semantic_teacher_image_edit_paid_authority(
            bundle_path=bundle,
            bundle_receipt_path=receipt_path,
            authorization_reference="fixture",
            authorized_by="fixture-user",
            authorized_on="2026-08-13",
            source_commit_sha=receipt["source_commit_sha"],
            backend_entry_digest=receipt["backend_entry_digest"],
            task_count=1,
            camera_count=2,
            maximum_hourly_rate_usd=0.5,
            hard_total_spend_cap_usd=1.0,
            hard_ttl_seconds=600,
            aggregate_goal_spend_before_attempt_usd=0.1,
            aggregate_goal_spend_cap_usd=20.0,
            output_path=tmp_path / "authority.json",
        )


def _prior_spend_reconciliation(
    tmp_path: Path, *, historical_gap: bool = False
) -> tuple[Path, float]:
    cost = 0.387839 if historical_gap else 0.1
    authority_digest = "sha256:" + "4" * 64
    bundle_digest = None if historical_gap else "sha256:" + "5" * 64
    terminal = {
        "schema_version": "fixture_prior_terminal.v1",
        "status": "completed",
        "cost_usd": cost,
        "continuing_spend_from_this_run": False,
        "run_instance_id": "prior-123",
        "authority_digest": authority_digest,
        "bundle_sha256": bundle_digest,
        "raw_secret_values_recorded": False,
        "result_digest": "",
    }
    terminal["result_digest"] = canonical_digest(terminal, digest_field="result_digest")
    zero = {
        "schema_version": "fixture_prior_zero.v1",
        "status": "provider_zero",
        "provider_zero_api_confirmed": True,
        "run_instance_ids": ["prior-123"],
        "raw_secret_values_recorded": False,
        "provider_zero_digest": "",
    }
    zero["provider_zero_digest"] = canonical_digest(
        zero, digest_field="provider_zero_digest"
    )
    terminal_path = tmp_path / "prior-terminal.json"
    zero_path = tmp_path / "prior-zero.json"
    _write_json(terminal_path, terminal)
    _write_json(zero_path, zero)
    source_receipts = [
        {
            "role": "terminal",
            "schema_version": terminal["schema_version"],
            "digest_field": "result_digest",
            "record": {
                **_record(terminal_path),
                "receipt_digest": terminal["result_digest"],
            },
        },
        {
            "role": "provider_zero",
            "schema_version": zero["schema_version"],
            "digest_field": "provider_zero_digest",
            "record": {
                **_record(zero_path),
                "receipt_digest": zero["provider_zero_digest"],
            },
        },
    ]
    bindings = [
        {"kind": "cost_usd", "source_role": "terminal", "json_path": ["cost_usd"], "expected_value": cost},
        {"kind": "continuing_spend", "source_role": "terminal", "json_path": ["continuing_spend_from_this_run"], "expected_value": False},
        {"kind": "instance_id", "source_role": "terminal", "json_path": ["run_instance_id"], "expected_value": "prior-123"},
        {"kind": "authority_digest", "source_role": "terminal", "json_path": ["authority_digest"], "expected_value": authority_digest},
        {"kind": "provider_zero", "source_role": "provider_zero", "json_path": ["provider_zero_api_confirmed"], "expected_value": True},
    ]
    if not historical_gap:
        bindings.append(
            {"kind": "bundle_sha256", "source_role": "terminal", "json_path": ["bundle_sha256"], "expected_value": bundle_digest}
        )
    entry = {
        "schema_version": "adp_same_goal_spend_entry.v1",
        "goal_id": "arm-decision-proof-v1",
        "attempt_id": "adp-ca-a-pan-chera-v8r2" if historical_gap else "prior-attempt",
        "lane": "content_agents_pan_chera_v8r2" if historical_gap else "fixture-lane",
        "evidence_kind": "historical_evidence_gap" if historical_gap else "fully_bound",
        "cost_usd": cost,
        "authority_digest": authority_digest,
        "bundle_sha256": bundle_digest,
        "continuing_spend_from_this_run": False,
        "provider_zero_confirmed": True,
        "source_receipts": source_receipts,
        "bindings": bindings,
        "entry_digest": "",
    }
    if historical_gap:
        entry.update(
            {
                "bundle_digest_available": False,
                "typed_abstention": "historical_bundle_bytes_not_retained",
            }
        )
    entry["entry_digest"] = canonical_digest(entry, digest_field="entry_digest")
    reconciliation = {
        "schema_version": "adp_same_goal_spend_reconciliation.v1",
        "status": "all_same_goal_paid_attempts_terminal_and_provider_zero",
        "goal_id": "arm-decision-proof-v1",
        "entries": [entry],
        "entry_count": 1,
        "total_cost_usd": cost,
        "receipt_digest": "",
    }
    reconciliation["receipt_digest"] = canonical_digest(
        reconciliation, digest_field="receipt_digest"
    )
    path = tmp_path / "prior-reconciliation.json"
    _write_json(path, reconciliation)
    return path, cost


@pytest.mark.parametrize("historical_gap", [False, True])
def test_authority_reopens_typed_prior_spend_entries(
    tmp_path: Path, historical_gap: bool
) -> None:
    receipt, bundle, receipt_path = _build(tmp_path, task_count=1)
    prior_path, prior_cost = _prior_spend_reconciliation(
        tmp_path, historical_gap=historical_gap
    )
    authority = materialize_semantic_teacher_image_edit_paid_authority(
        bundle_path=bundle,
        bundle_receipt_path=receipt_path,
        authorization_reference="fixture",
        authorized_by="fixture-user",
        authorized_on="2026-08-13",
        source_commit_sha=receipt["source_commit_sha"],
        backend_entry_digest=receipt["backend_entry_digest"],
        task_count=1,
        camera_count=2,
        maximum_hourly_rate_usd=0.5,
        hard_total_spend_cap_usd=1.0,
        hard_ttl_seconds=600,
        aggregate_goal_spend_before_attempt_usd=prior_cost,
        aggregate_goal_spend_cap_usd=20.0,
        prior_spend_reconciliation_path=prior_path,
        output_path=tmp_path / "authority.json",
    )
    assert authority["prior_spend_reconciliation"]["entry_count"] == 1
    assert authority["aggregate_goal_spend_before_attempt_usd"] == prior_cost


def test_authority_is_atomically_single_use(tmp_path: Path, monkeypatch) -> None:
    authority, _path, _bundle, _receipt, _receipt_path = _authority(tmp_path)
    root = tmp_path / "consumption-root"

    def prepare_root() -> Path:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        return root

    monkeypatch.setattr(
        "blueprint_pipeline.semantic_teacher_image_edit_paid_authority.prepare_consumption_root",
        prepare_root,
    )
    first = consume_semantic_teacher_image_edit_paid_authority_once(
        authority, source_commit_sha=authority["source_commit_sha"]
    )
    second = consume_semantic_teacher_image_edit_paid_authority_once(
        authority, source_commit_sha=authority["source_commit_sha"]
    )
    assert first["status"] == "consumed"
    assert second == {
        "status": "blocked",
        "blockers": ["semantic_teacher_paid_authority_consumed"],
    }
    records = list(root.glob("semantic-teacher-*.json"))
    assert len(records) == 1
    assert records[0].stat().st_mode & 0o077 == 0


def test_authority_consumption_rejects_digest_or_commit_tamper(tmp_path: Path) -> None:
    authority, _path, _bundle, _receipt, _receipt_path = _authority(tmp_path)
    tampered = dict(authority)
    tampered["camera_count"] = 5
    assert consume_semantic_teacher_image_edit_paid_authority_once(
        tampered, source_commit_sha=authority["source_commit_sha"]
    )["status"] == "blocked"
    assert consume_semantic_teacher_image_edit_paid_authority_once(
        authority, source_commit_sha="f" * 40
    )["status"] == "blocked"


def _inventory(instance_id: str | None = None) -> dict:
    value = {
        "provider": "vast",
        "api_confirmed": True,
        "live_resource_count": 0,
        "resources": [],
    }
    if instance_id is not None:
        value["queried_instance_ids"] = [instance_id]
        value["absent_instance_ids"] = [instance_id]
    return value


def test_allocator_dry_run_and_no_allocation_closeout_are_zero_mutation(
    tmp_path: Path,
) -> None:
    _authority_value, authority_path, bundle, _receipt, receipt_path = _authority(tmp_path)
    dry_path = tmp_path / "dry-run.json"
    dry = prepare_semantic_teacher_image_edit_allocator_dry_run(
        authority_path=authority_path,
        bundle_path=bundle,
        bundle_receipt_path=receipt_path,
        checkout_source_commit=_authority_value["source_commit_sha"],
        live_inventory=_inventory(),
        output_path=dry_path,
    )
    assert dry["status"] == "dry_run_ready"
    assert dry["token_lookup_performed"] is False
    assert dry["object_store_staging_performed"] is False
    assert dry["provider_mutations_performed"] == 0
    watchdog_path = tmp_path / "watchdog-no-allocation.json"
    zero = _inventory()
    _write_json(
        watchdog_path,
        {
            "schema_version": "vast_independent_watchdog_handoff.v1",
            "status": "provider_terminal",
            "watchdog_armed_before_allocation": True,
            "provider_absence_confirmed": True,
            "initial_inventory": zero,
            "initial_global_inventory": zero,
            "final_inventory": zero,
            "final_global_inventory": zero,
            "provider_mutations_performed": 0,
            "raw_secret_values_recorded": False,
        },
    )
    closeout = materialize_semantic_teacher_no_allocation_closeout(
        dry_run_path=dry_path,
        watchdog_closeout_path=watchdog_path,
        reason="capacity_not_admitted",
        output_path=tmp_path / "no-allocation.json",
    )
    assert closeout["allocation_count"] == 0
    assert closeout["all_staged_objects_absent"] is True
    assert closeout["continuing_spend_from_this_run"] is False


def test_allocator_dry_run_requires_api_confirmed_zero(tmp_path: Path) -> None:
    _authority_value, authority_path, bundle, _receipt, receipt_path = _authority(tmp_path)
    with pytest.raises(
        ValueError, match="semantic_teacher_dry_run_provider_inventory_not_zero"
    ):
        prepare_semantic_teacher_image_edit_allocator_dry_run(
            authority_path=authority_path,
            bundle_path=bundle,
            bundle_receipt_path=receipt_path,
            checkout_source_commit=_authority_value["source_commit_sha"],
            live_inventory={
                "api_confirmed": True,
                "live_resource_count": 1,
                "resources": [{"id": "run-owned"}],
            },
            output_path=tmp_path / "dry-run.json",
        )


def _terminal_fixture(tmp_path: Path) -> dict[str, Path | dict]:
    authority, authority_path, bundle_path, bundle_receipt, bundle_receipt_path = _authority(
        tmp_path
    )
    runtime_request_path = tmp_path / "runtime-request.json"
    with zipfile.ZipFile(bundle_path) as archive:
        runtime_request_path.write_bytes(
            archive.read(
                "provider_runtime/semantic_teacher_image_edit_runtime_request.v1.json"
            )
        )
    runtime_request = json.loads(runtime_request_path.read_text())
    runtime = tmp_path / "runtime-output"
    frame_paths: list[Path] = []
    task_rows = []
    for task_id in ("task_a", "task_b"):
        path = runtime / "tasks" / task_id / "00000.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), (1, 2, 3)).save(path)
        frame_paths.append(path)
        task_rows.append(
            {
                "task_id": task_id,
                "camera_count": 1,
                "frames": [
                    {
                        "frame_index": 0,
                        "semantic_teacher_frame": _record(path, root=runtime),
                    }
                ],
            }
        )
    runtime_result = {
        "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
        "status": "completed_unreviewed_semantic_teacher_candidates",
        "task_count": 2,
        "request_count": 2,
        "attempted_request_count": 2,
        "successful_request_count": 2,
        "source_runtime_request_digest": runtime_request["request_digest"],
        "backend_entry_digest": authority["backend_entry_digest"],
        "retry_count": 0,
        "tasks": task_rows,
        "raw_secret_values_recorded": False,
        "result_digest": "",
    }
    runtime_result["result_digest"] = canonical_digest(
        runtime_result, digest_field="result_digest"
    )
    terminal_path = runtime / f"{RUNTIME_RESULT_SCHEMA_VERSION}.json"
    _write_json(terminal_path, runtime_result)
    stdout = runtime / "runtime_stdout.log"
    stderr = runtime / "runtime_stderr.log"
    stdout.write_text("fixture stdout\n", encoding="utf-8")
    stderr.write_text("fixture stderr\n", encoding="utf-8")
    instance_id = "123"
    billing = tmp_path / "billing.json"
    cleanup = tmp_path / "cleanup.json"
    watchdog = tmp_path / "watchdog.json"
    scoped = tmp_path / "scoped-inventory.json"
    global_inventory = tmp_path / "global-inventory.json"
    redaction = tmp_path / "redaction.json"
    billing_value = {
        "schema_version": "semantic_teacher_image_edit_billing.v1",
        "status": "completed",
        "authority_digest": authority["authorization_digest"],
        "bundle_sha256": authority["bundle"]["sha256"],
        "runtime_request_digest": authority["runtime_request_digest"],
        "backend_entry_digest": authority["backend_entry_digest"],
        "pricing_binding_digest": authority["pricing_binding_digest"],
        "run_instance_id": instance_id,
        "attempted_request_count": 2,
        "maximum_cost_per_request_usd": authority["maximum_cost_per_request_usd"],
        "editor_request_cost_usd": 0.2,
        "compute_cost_usd": 0.05,
        "cost_usd": 0.25,
        "raw_secret_values_recorded": False,
        "billing_digest": "",
    }
    billing_value["billing_digest"] = canonical_digest(
        billing_value, digest_field="billing_digest"
    )
    _write_json(billing, billing_value)
    _write_json(
        cleanup,
        {
            "schema_version": "wam_provider_object_store_cleanup.v1",
            "status": "completed",
            "all_objects_absent": True,
            "signed_url_files_removed": True,
            "blockers": [],
            "raw_secret_values_recorded": False,
        },
    )
    _write_json(
        watchdog,
        {
            "schema_version": "vast_independent_watchdog_handoff.v1",
            "status": "provider_terminal",
            "watchdog_armed_before_allocation": True,
            "provider_absence_confirmed": True,
            "instance_ids": [instance_id],
            "raw_secret_values_recorded": False,
        },
    )
    _write_json(scoped, _inventory(instance_id))
    _write_json(global_inventory, _inventory())
    redaction_value = {
        "schema_version": "semantic_teacher_image_edit_secret_redaction.v1",
        "status": "passed",
        "authority_digest": authority["authorization_digest"],
        "bundle_sha256": authority["bundle"]["sha256"],
        "run_instance_id": instance_id,
        "stdout_sha256": _sha256(stdout),
        "stderr_sha256": _sha256(stderr),
        "stdout_scanned": True,
        "stderr_scanned": True,
        "secret_values_found": False,
        "raw_secret_values_recorded": False,
        "redaction_digest": "",
    }
    redaction_value["redaction_digest"] = canonical_digest(
        redaction_value, digest_field="redaction_digest"
    )
    _write_json(redaction, redaction_value)
    provider_zero = tmp_path / "provider-zero.json"
    materialize_semantic_teacher_provider_zero_receipt(
        authority_path=authority_path,
        bundle_receipt_path=bundle_receipt_path,
        terminal_result_path=terminal_path,
        billing_receipt_path=billing,
        scoped_inventory_path=scoped,
        global_inventory_path=global_inventory,
        object_store_cleanup_path=cleanup,
        independent_watchdog_path=watchdog,
        secret_redaction_path=redaction,
        stdout_log_path=stdout,
        stderr_log_path=stderr,
        output_path=provider_zero,
    )
    return {
        "authority": authority,
        "authority_path": authority_path,
        "bundle_receipt": bundle_receipt,
        "bundle_receipt_path": bundle_receipt_path,
        "runtime_request_path": runtime_request_path,
        "runtime": runtime,
        "terminal_path": terminal_path,
        "billing": billing,
        "cleanup": cleanup,
        "watchdog": watchdog,
        "scoped": scoped,
        "global": global_inventory,
        "redaction": redaction,
        "provider_zero": provider_zero,
        "frame_paths": frame_paths,
    }


def test_provider_zero_requires_cleanup_watchdog_logs_and_dual_api_zero(
    tmp_path: Path,
) -> None:
    fixture = _terminal_fixture(tmp_path)
    receipt = json.loads(Path(fixture["provider_zero"]).read_text())
    assert receipt["provider_zero_api_confirmed"] is True
    assert receipt["all_staged_objects_absent"] is True
    assert receipt["continuing_spend_from_this_run"] is False
    cleanup = Path(fixture["cleanup"])
    cleanup_value = json.loads(cleanup.read_text())
    cleanup_value["signed_url_files_removed"] = False
    _write_json(cleanup, cleanup_value)
    with pytest.raises(ValueError, match="semantic_teacher_provider_zero_inputs_invalid"):
        materialize_semantic_teacher_provider_zero_receipt(
            authority_path=fixture["authority_path"],
            bundle_receipt_path=fixture["bundle_receipt_path"],
            terminal_result_path=fixture["terminal_path"],
            billing_receipt_path=fixture["billing"],
            scoped_inventory_path=fixture["scoped"],
            global_inventory_path=fixture["global"],
            object_store_cleanup_path=cleanup,
            independent_watchdog_path=fixture["watchdog"],
            secret_redaction_path=fixture["redaction"],
            stdout_log_path=Path(fixture["runtime"]) / "runtime_stdout.log",
            stderr_log_path=Path(fixture["runtime"]) / "runtime_stderr.log",
            output_path=tmp_path / "invalid-provider-zero.json",
        )


def _runtime_gap_closeout(
    tmp_path: Path,
    *,
    mutation: str | None = None,
) -> tuple[dict, dict]:
    fixture = _terminal_fixture(tmp_path)
    authority = fixture["authority"]
    assert isinstance(authority, dict)
    instance_id = "123"
    gap = {
        "schema_version": "semantic_teacher_image_edit_runtime_media_gap.v1",
        "status": "blocked_runtime_result_missing",
        "gap_type": "runtime_timeout",
        "reason_code": "output_download_timed_out_after_allocation",
        "authority_digest": authority["authorization_digest"],
        "bundle_sha256": authority["bundle"]["sha256"],
        "runtime_request_digest": authority["runtime_request_digest"],
        "backend_entry_digest": authority["backend_entry_digest"],
        "run_instance_id": instance_id,
        "attempted_request_count_known": False,
        "attempted_request_count_upper_bound": authority["camera_count"],
        "runtime_result_present": False,
        "media_complete": False,
        "teacher_frames_qualified": False,
        "partial_png_inventory": [],
        "raw_secret_values_recorded": False,
        "gap_digest": "",
    }
    billing_path = Path(fixture["billing"])
    billing = json.loads(billing_path.read_text())
    billing.update(
        {
            "status": "conservative_upper_bound_runtime_result_missing",
            "attempted_request_count_known": False,
            "attempted_request_count_upper_bound": authority["camera_count"],
            "editor_request_cost_usd": authority[
                "hosted_editor_spend_upper_bound_usd"
            ],
            "editor_request_cost_basis": (
                "full_authorized_upper_bound_due_to_unknown_attempt_count"
            ),
            "cost_usd": authority["hosted_editor_spend_upper_bound_usd"]
            + billing["compute_cost_usd"],
        }
    )
    billing.pop("attempted_request_count", None)
    if mutation == "fabricated_attempt_count":
        billing["attempted_request_count"] = 0
    elif mutation == "undercharged_hosted_editor":
        billing["editor_request_cost_usd"] = 0.0
        billing["cost_usd"] = billing["compute_cost_usd"]
    elif mutation == "wrong_attempt_upper_bound":
        billing["attempted_request_count_upper_bound"] -= 1
    elif mutation == "gap_fabricated_attempt_count":
        gap["attempted_request_count"] = 0
    billing["billing_digest"] = canonical_digest(
        billing, digest_field="billing_digest"
    )
    gap["gap_digest"] = canonical_digest(gap, digest_field="gap_digest")
    gap_path = tmp_path / "runtime-media-gap.json"
    _write_json(gap_path, gap)
    _write_json(billing_path, billing)
    output_path = tmp_path / "gap-provider-zero.json"
    result = materialize_semantic_teacher_provider_zero_receipt(
        authority_path=fixture["authority_path"],
        bundle_receipt_path=fixture["bundle_receipt_path"],
        terminal_result_path=gap_path,
        billing_receipt_path=billing_path,
        scoped_inventory_path=fixture["scoped"],
        global_inventory_path=fixture["global"],
        object_store_cleanup_path=fixture["cleanup"],
        independent_watchdog_path=fixture["watchdog"],
        secret_redaction_path=fixture["redaction"],
        stdout_log_path=Path(fixture["runtime"]) / "runtime_stdout.log",
        stderr_log_path=Path(fixture["runtime"]) / "runtime_stderr.log",
        output_path=output_path,
    )
    return result, fixture


def test_allocated_missing_runtime_result_retains_conservative_zero_close(
    tmp_path: Path,
) -> None:
    result, fixture = _runtime_gap_closeout(tmp_path)
    authority = fixture["authority"]
    assert isinstance(authority, dict)
    assert result["terminal_evidence_kind"] == "runtime_media_gap"
    assert result["terminal_result_digest"] is None
    assert result["runtime_media_gap_digest"].startswith("sha256:")
    assert result["attempted_request_count_known"] is False
    assert result["attempted_request_count_upper_bound"] == authority["camera_count"]
    assert result["total_cost_usd"] == pytest.approx(
        authority["hosted_editor_spend_upper_bound_usd"] + 0.05
    )
    assert result["provider_zero_api_confirmed"] is True
    assert result["continuing_spend_from_this_run"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "fabricated_attempt_count",
        "undercharged_hosted_editor",
        "wrong_attempt_upper_bound",
        "gap_fabricated_attempt_count",
    ],
)
def test_allocated_missing_runtime_result_rejects_fabricated_or_undercharged_truth(
    tmp_path: Path,
    mutation: str,
) -> None:
    with pytest.raises(
        ValueError, match="semantic_teacher_provider_zero_inputs_invalid"
    ):
        _runtime_gap_closeout(tmp_path, mutation=mutation)


def test_result_import_retains_every_png_log_and_typed_closeout(tmp_path: Path) -> None:
    fixture = _terminal_fixture(tmp_path)
    imported = materialize_semantic_teacher_image_edit_result(
        runtime_output_root=fixture["runtime"],
        runtime_request_path=fixture["runtime_request_path"],
        bundle_receipt_path=fixture["bundle_receipt_path"],
        authority_path=fixture["authority_path"],
        billing_receipt_path=fixture["billing"],
        scoped_inventory_path=fixture["scoped"],
        global_inventory_path=fixture["global"],
        object_store_cleanup_path=fixture["cleanup"],
        watchdog_receipt_path=fixture["watchdog"],
        secret_redaction_path=fixture["redaction"],
        provider_zero_path=fixture["provider_zero"],
        expected_task_count=2,
        expected_camera_count=2,
        output_path=tmp_path / "import.json",
    )
    assert imported["all_generated_teacher_pngs_retained"] is True
    assert len(imported["teacher_frames"]) == len(fixture["frame_paths"])
    assert imported["runtime_stdout"]["size_bytes"] > 0
    assert imported["visual_reviewed"] is False
    assert imported["appearance_qualified"] is False


def test_result_import_rejects_unallowlisted_output(tmp_path: Path) -> None:
    fixture = _terminal_fixture(tmp_path)
    runtime = Path(fixture["runtime"])
    (runtime / "unbound.bin").write_bytes(b"unbound")
    with pytest.raises(ValueError, match="semantic_teacher_result_file_not_allowlisted"):
        materialize_semantic_teacher_image_edit_result(
            runtime_output_root=runtime,
            runtime_request_path=fixture["runtime_request_path"],
            bundle_receipt_path=fixture["bundle_receipt_path"],
            authority_path=fixture["authority_path"],
            billing_receipt_path=fixture["billing"],
            scoped_inventory_path=fixture["scoped"],
            global_inventory_path=fixture["global"],
            object_store_cleanup_path=fixture["cleanup"],
            watchdog_receipt_path=fixture["watchdog"],
            secret_redaction_path=fixture["redaction"],
            provider_zero_path=fixture["provider_zero"],
            expected_task_count=2,
            expected_camera_count=2,
            output_path=tmp_path / "import.json",
        )
