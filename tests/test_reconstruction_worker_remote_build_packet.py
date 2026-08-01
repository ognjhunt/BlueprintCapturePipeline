from __future__ import annotations

import json
import subprocess
import tarfile
from datetime import date
from pathlib import Path

from jsonschema import Draft202012Validator

from blueprint_pipeline.reconstruction_worker_build_packet import (
    BUILD_SCRIPT_NAME,
    DEFAULT_IMAGE_REF,
    LICENSE_INVENTORY_NAME,
    LICENSE_RECEIPT_NAME,
    PAID_ENVELOPE_NAME,
    PACKET_DIRNAME,
    prepare_reconstruction_worker_remote_build_packet,
    validate_reconstruction_worker_archive,
)
from blueprint_pipeline.reconstruction_worker_contracts import (
    PINNED_MODEL_ASSETS,
    build_worker_stack_manifest,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.groot_oscar_infrastructure_admission import (
    MIN_BUILD_FREE_BYTES,
    build_build_plane_admission,
)
from blueprint_pipeline.image_build_result_verification import (
    validate_remote_reconstruction_worker_result,
)
from blueprint_pipeline.reconstruction_worker_license_inventory import (
    build_reconstruction_worker_license_inventory,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _fixture_repo(root: Path) -> str:
    files = {
        "deploy/docker/reconstruction_worker/Dockerfile": "FROM scratch\n",
        "deploy/docker/reconstruction_worker/build-requirements.in": "pip==24.2\n",
        "deploy/docker/reconstruction_worker/build-requirements.lock": "pip==24.2\n",
        "deploy/docker/reconstruction_worker/requirements.in": "numpy==1.26.4\n",
        "deploy/docker/reconstruction_worker/requirements.lock": "numpy==1.26.4\n",
        "scripts/compile_reconstruction_worker_lock.py": "print('fixture')\n",
        "pyproject.toml": "[project]\nname='fixture'\nversion='1.0.0'\n",
        "README.md": "fixture\n",
        "LICENSE": "fixture license\n",
        "src/blueprint_pipeline/__init__.py": "\n",
        "src/blueprint_pipeline/worker.py": "VALUE = 1\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _worker_stack(source_commit: str) -> dict:
    return build_worker_stack_manifest(
        {
            "worker_family": "blueprint-reconstruction-worker",
            "runnable_platform": "linux/amd64",
            "headless_required": True,
            "display_required": False,
            "source_commit_sha": source_commit,
            "qualification_status": "candidate_unbuilt",
            "minimum_vram_gb": 24,
            "supported_compute_capabilities": [75, 80, 86, 89],
            "tested_driver_range": {"status": "not_yet_tested"},
            "model_assets": list(PINNED_MODEL_ASSETS),
            "hidden_heldout_access": False,
            "trainer_self_grading": False,
        }
    )


def _license_inventory(source_commit: str) -> dict:
    return build_reconstruction_worker_license_inventory(
        source_commit_sha=source_commit,
        worker_stack_manifest=_worker_stack(source_commit),
        requirements_lock_path=(
            REPO_ROOT / "deploy/docker/reconstruction_worker/requirements.lock"
        ),
        license_policy=json.loads(
            (REPO_ROOT / "docs/runtime_dependency_license_policy.json").read_text(
                encoding="utf-8"
            )
        ),
        as_of=date(2026, 8, 1),
    )


def _license_receipt(source_commit: str, inventory: dict | None = None) -> dict:
    inventory = inventory or _license_inventory(source_commit)
    value = {
        "schema_version": "reconstruction_worker_license_review_receipt.v2",
        "status": "accepted_internal_build_only",
        "source_commit_sha": source_commit,
        "worker_stack_manifest_digest": inventory["worker_stack_manifest_digest"],
        "requirements_lock_digest": inventory["requirements_lock_digest"],
        "license_inventory_digest": inventory["license_inventory_digest"],
        "license_policy_digest": inventory["license_policy_digest"],
        "registry_visibility": "private",
        "internal_build_authorized": True,
        "redistribution_authorized": False,
        "commercial_distribution_authorized": False,
        "review_basis": "human_review_of_digest_bound_inventory",
        "reviewer_authority_id": "fixture-legal-reviewer",
        "reviewed_dependency_count": inventory["dependency_count"],
        "reviewed_source_component_ids": sorted(
            row["component_id"] for row in inventory["source_component_reviews"]
        ),
        "reviewed_model_asset_ids": sorted(
            row["model_id"] for row in inventory["model_asset_reviews"]
        ),
        "acknowledged_inventory_blockers": inventory["blockers"],
        "timestamp": "2026-08-01T12:00:00Z",
        "warnings": ["fixture receipt only"],
    }
    value["license_review_receipt_digest"] = canonical_digest(
        value, digest_field="license_review_receipt_digest"
    )
    return value


def _paid_envelope(
    source_commit: str, inventory: dict, license_receipt: dict
) -> dict:
    value = {
        "schema_version": "reconstruction_worker_paid_execution_envelope.v1",
        "authorized_action": "cpu-build",
        "paid_mutation_authorized": True,
        "authority_issued_by_agent": False,
        "authority_id": "fixture-paid-authority",
        "max_spend_usd": 1.0,
        "hard_ttl_seconds": 3600,
        "retry_cap": 0,
        "source_commit_sha": source_commit,
        "worker_stack_manifest_digest": inventory["worker_stack_manifest_digest"],
        "license_inventory_digest": inventory["license_inventory_digest"],
        "license_review_receipt_digest": license_receipt[
            "license_review_receipt_digest"
        ],
    }
    value["paid_execution_envelope_digest"] = canonical_digest(
        value, digest_field="paid_execution_envelope_digest"
    )
    return value


def test_remote_packet_is_clean_bound_deterministic_and_secret_free(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    head = _fixture_repo(repo)
    inventory = _license_inventory(head)
    receipt = _license_receipt(head, inventory)
    envelope = _paid_envelope(head, inventory, receipt)
    first = prepare_reconstruction_worker_remote_build_packet(
        output_dir=tmp_path / "first",
        repo_root=repo,
        image_ref=DEFAULT_IMAGE_REF,
        source_commit=head,
        source_worktree_dirty=False,
        worker_stack_manifest=_worker_stack(head),
        license_inventory=inventory,
        license_review_receipt=receipt,
        paid_execution_envelope=envelope,
        generated_at="2026-07-30T12:00:00Z",
    )
    second = prepare_reconstruction_worker_remote_build_packet(
        output_dir=tmp_path / "second",
        repo_root=repo,
        image_ref=DEFAULT_IMAGE_REF,
        source_commit=head,
        source_worktree_dirty=False,
        worker_stack_manifest=_worker_stack(head),
        license_inventory=inventory,
        license_review_receipt=receipt,
        paid_execution_envelope=envelope,
        generated_at="2026-07-30T12:00:00Z",
    )

    assert first["status"] == "ready"
    assert first["blockers"] == []
    assert first["tarball_sha256"] == second["tarball_sha256"]
    assert first["archive_member_sha256"] == second["archive_member_sha256"]
    assert validate_reconstruction_worker_archive(first) == []
    assert (
        f"{PACKET_DIRNAME}/context/deploy/docker/reconstruction_worker/build-requirements.in"
        in first["archive_members"]
    )
    assert (
        f"{PACKET_DIRNAME}/context/deploy/docker/reconstruction_worker/build-requirements.lock"
        in first["archive_members"]
    )
    assert f"{PACKET_DIRNAME}/context/deploy/docker/reconstruction_worker/requirements.in" in first[
        "archive_members"
    ]
    assert f"{PACKET_DIRNAME}/context/scripts/compile_reconstruction_worker_lock.py" in first[
        "archive_members"
    ]
    assert f"{PACKET_DIRNAME}/{LICENSE_INVENTORY_NAME}" in first["archive_members"]
    assert f"{PACKET_DIRNAME}/{LICENSE_RECEIPT_NAME}" in first["archive_members"]
    assert f"{PACKET_DIRNAME}/{PAID_ENVELOPE_NAME}" in first["archive_members"]
    assert first["provider_launch_performed_by_packet"] is False
    assert first["raw_secret_values_recorded"] is False
    assert not any(
        name.endswith((".env", ".pem", ".key")) or "/.git/" in name
        for name in first["archive_members"]
    )
    with tarfile.open(first["tarball_path"], "r:gz") as archive:
        script = archive.extractfile(f"{PACKET_DIRNAME}/{BUILD_SCRIPT_NAME}")
        assert script is not None
        source = script.read().decode("utf-8")
    assert "docker buildx build" in source
    assert "--platform linux/amd64" in source
    assert "--provenance=true" in source
    assert "--sbom=true" in source
    assert 'rm -f "$username_file"' not in source

    schema_path = (
        Path(__file__).resolve().parents[1]
        / "docs/schemas/reconstruction_worker_remote_build.v2.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    recorded = json.loads(Path(first["manifest_path"]).read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(recorded)
    license_schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/schemas/reconstruction_worker_license_review_receipt.v2.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator(license_schema).validate(_license_receipt(head, inventory))
    paid_schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/schemas/reconstruction_worker_paid_execution_envelope.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator(paid_schema).validate(envelope)


def test_remote_packet_archive_tamper_fails_closed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    head = _fixture_repo(repo)
    inventory = _license_inventory(head)
    receipt = _license_receipt(head, inventory)
    packet = prepare_reconstruction_worker_remote_build_packet(
        output_dir=tmp_path / "packet",
        repo_root=repo,
        image_ref=DEFAULT_IMAGE_REF,
        source_commit=head,
        source_worktree_dirty=False,
        worker_stack_manifest=_worker_stack(head),
        license_inventory=inventory,
        license_review_receipt=receipt,
        paid_execution_envelope=_paid_envelope(head, inventory, receipt),
    )
    path = Path(packet["tarball_path"])
    path.write_bytes(path.read_bytes() + b"tamper")

    assert "builder_reconstruction_archive_tarball_mismatch" in (
        validate_reconstruction_worker_archive(packet)
    )


def test_remote_packet_enters_canonical_cpu_build_admission_and_result_verifier(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    head = _fixture_repo(repo)
    inventory = _license_inventory(head)
    receipt = _license_receipt(head, inventory)
    packet = prepare_reconstruction_worker_remote_build_packet(
        output_dir=tmp_path / "packet",
        repo_root=repo,
        image_ref=DEFAULT_IMAGE_REF,
        source_commit=head,
        source_worktree_dirty=False,
        worker_stack_manifest=_worker_stack(head),
        license_inventory=inventory,
        license_review_receipt=receipt,
        paid_execution_envelope=_paid_envelope(head, inventory, receipt),
    )
    admission = build_build_plane_admission(
        packet=packet,
        builder={
            "provider": "github_actions",
            "purpose": "image_build",
            "platform": "linux/amd64",
            "docker_daemon_verified": True,
            "docker_buildx_verified": True,
            "free_disk_bytes": MIN_BUILD_FREE_BYTES,
            "registry_push_auth_file_verified": True,
            "independent_teardown_watchdog": True,
            "expected_source_commit": head,
        },
        spend={
            "paid_mutation_authorized": True,
            "max_spend_usd": 1.0,
            "hard_ttl_seconds": 3600,
            "retry_cap": 0,
            "authority_id": "fixture-paid-authority",
            "authority_issued_by_agent": False,
            "one_resource_limit": True,
        },
    )
    assert admission["status"] == "admitted"
    assert admission["checks"]["packet_kind"] == "reconstruction_worker_image"
    drifted_admission = build_build_plane_admission(
        packet=packet,
        builder={
            "provider": "github_actions",
            "purpose": "image_build",
            "platform": "linux/amd64",
            "docker_daemon_verified": True,
            "docker_buildx_verified": True,
            "free_disk_bytes": MIN_BUILD_FREE_BYTES,
            "registry_push_auth_file_verified": True,
            "independent_teardown_watchdog": True,
            "expected_source_commit": head,
        },
        spend={
            "paid_mutation_authorized": True,
            "max_spend_usd": 0.5,
            "hard_ttl_seconds": 3600,
            "retry_cap": 0,
            "authority_id": "fixture-paid-authority",
            "authority_issued_by_agent": False,
            "one_resource_limit": True,
        },
    )
    assert drifted_admission["status"] == "blocked"
    assert (
        "builder_reconstruction_spend_envelope_mismatch"
        in drifted_admission["blockers"]
    )

    receipt = {
        "schema_version": "reconstruction_worker_build_receipt.v2",
        "timestamp": "2026-07-30T12:00:00Z",
        "worker_stack_manifest_digest": packet["worker_stack_manifest_digest"],
        "license_inventory_digest": packet["license_inventory_digest"],
        "license_review_receipt_digest": packet["license_review_receipt_digest"],
        "paid_execution_envelope_digest": packet[
            "paid_execution_envelope_digest"
        ],
        "status": "built",
        "resolved_image_digest": packet["image_ref"].rsplit(":", 1)[0]
        + "@sha256:"
        + "a" * 64,
        "source_commit_sha": packet["source_commit"],
        "build_context_digest": "sha256:" + packet["context_manifest_sha256"],
        "duration_seconds": 12.0,
        "cost_usd": 0.0,
        "logs": ["reconstruction_worker_build_metadata.json"],
        "blockers": [],
        "scientific_qualification_inferred": False,
        "build_healthcheck_embedded": True,
        "runtime_gpu_healthcheck_completed": False,
        "raw_secret_values_recorded": False,
        "proof_effect": "none",
        "claim_ceiling": "resolved_worker_image_build_only",
    }
    receipt["build_receipt_digest"] = canonical_digest(
        receipt, digest_field="build_receipt_digest"
    )
    result_path = tmp_path / "results/reconstruction_worker_build_receipt.json"
    result_path.parent.mkdir()
    result_path.write_text(json.dumps(receipt), encoding="utf-8")
    verified = validate_remote_reconstruction_worker_result(
        result_path.parent, packet=packet
    )
    assert verified["status"] == "verified"
    assert verified["proof_effect"] == "none"
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "docs/schemas/reconstruction_worker_remote_build.v2.schema.json"
    )
    Draft202012Validator(
        json.loads(schema_path.read_text(encoding="utf-8"))
    ).validate(receipt)


def test_remote_packet_rejects_dirty_or_spoofed_source_identity(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    head = _fixture_repo(repo)
    inventory = _license_inventory(head)
    receipt = _license_receipt(head, inventory)
    (repo / "README.md").write_text("dirty\n", encoding="utf-8")
    packet = prepare_reconstruction_worker_remote_build_packet(
        output_dir=tmp_path / "packet",
        repo_root=repo,
        image_ref=DEFAULT_IMAGE_REF,
        source_commit="f" * 40,
        source_worktree_dirty=False,
        worker_stack_manifest=_worker_stack(head),
        license_inventory=inventory,
        license_review_receipt=receipt,
        paid_execution_envelope=_paid_envelope(head, inventory, receipt),
    )

    assert packet["status"] == "blocked"
    assert {
        "reconstruction_worker_source_commit_not_exact_head",
        "reconstruction_worker_source_dirty_claim_mismatch",
        "reconstruction_worker_packet_requires_clean_source_worktree",
    } <= set(packet["blockers"])
    assert head != packet["source_commit"]


def test_remote_packet_cannot_infer_license_review_authority(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    head = _fixture_repo(repo)
    inventory = _license_inventory(head)
    receipt = _license_receipt(head, inventory)
    packet = prepare_reconstruction_worker_remote_build_packet(
        output_dir=tmp_path / "packet",
        repo_root=repo,
        image_ref=DEFAULT_IMAGE_REF,
        source_commit=head,
        source_worktree_dirty=False,
        worker_stack_manifest=_worker_stack(head),
        license_inventory=inventory,
        license_review_receipt=None,
        paid_execution_envelope=_paid_envelope(head, inventory, receipt),
    )

    assert packet["status"] == "blocked"
    assert "reconstruction_worker_license_review_receipt_missing" in packet["blockers"]


def test_remote_packet_requires_inventory_even_with_a_review_receipt(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    head = _fixture_repo(repo)
    inventory = _license_inventory(head)
    receipt = _license_receipt(head, inventory)
    packet = prepare_reconstruction_worker_remote_build_packet(
        output_dir=tmp_path / "packet",
        repo_root=repo,
        image_ref=DEFAULT_IMAGE_REF,
        source_commit=head,
        source_worktree_dirty=False,
        worker_stack_manifest=_worker_stack(head),
        license_inventory=None,
        license_review_receipt=receipt,
        paid_execution_envelope=_paid_envelope(head, inventory, receipt),
    )

    assert packet["status"] == "blocked"
    assert "reconstruction_worker_license_inventory_missing" in packet["blockers"]


def test_remote_packet_requires_non_agent_paid_execution_envelope(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    head = _fixture_repo(repo)
    inventory = _license_inventory(head)
    receipt = _license_receipt(head, inventory)
    packet = prepare_reconstruction_worker_remote_build_packet(
        output_dir=tmp_path / "packet",
        repo_root=repo,
        image_ref=DEFAULT_IMAGE_REF,
        source_commit=head,
        source_worktree_dirty=False,
        worker_stack_manifest=_worker_stack(head),
        license_inventory=inventory,
        license_review_receipt=receipt,
        paid_execution_envelope=None,
    )
    assert packet["status"] == "blocked"
    assert (
        "reconstruction_worker_paid_execution_envelope_missing"
        in packet["blockers"]
    )

    envelope = _paid_envelope(head, inventory, receipt)
    envelope["authority_issued_by_agent"] = True
    envelope["paid_execution_envelope_digest"] = canonical_digest(
        envelope, digest_field="paid_execution_envelope_digest"
    )
    packet = prepare_reconstruction_worker_remote_build_packet(
        output_dir=tmp_path / "agent-packet",
        repo_root=repo,
        image_ref=DEFAULT_IMAGE_REF,
        source_commit=head,
        source_worktree_dirty=False,
        worker_stack_manifest=_worker_stack(head),
        license_inventory=inventory,
        license_review_receipt=receipt,
        paid_execution_envelope=envelope,
    )
    assert packet["status"] == "blocked"
    assert (
        "reconstruction_worker_paid_envelope_agent_authority_forbidden"
        in packet["blockers"]
    )
