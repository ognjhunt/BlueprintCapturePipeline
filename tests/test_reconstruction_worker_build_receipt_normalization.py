from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.reconstruction_worker_build_packet import (
    DEFAULT_IMAGE_REF,
    prepare_reconstruction_worker_remote_build_packet,
)
from blueprint_pipeline.reconstruction_worker_build_receipt_normalization import (
    ReconstructionWorkerBuildNormalizationError,
    compile_reconstruction_worker_build_receipt,
)
from blueprint_pipeline.reconstruction_worker_contracts import (
    PINNED_MODEL_ASSETS,
    build_worker_stack_manifest,
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


def _stack(source_commit: str) -> dict:
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


def _inventory(source_commit: str, stack: dict) -> dict:
    return build_reconstruction_worker_license_inventory(
        source_commit_sha=source_commit,
        worker_stack_manifest=stack,
        requirements_lock_path=REPO_ROOT / "deploy/docker/reconstruction_worker/requirements.lock",
        license_policy=json.loads(
            (REPO_ROOT / "docs/runtime_dependency_license_policy.json").read_text()
        ),
        as_of=date(2026, 8, 1),
    )


def _sign(value: dict, field: str) -> dict:
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _inputs(tmp_path: Path) -> dict:
    repo = tmp_path / "repo"
    repo.mkdir()
    source_commit = _fixture_repo(repo)
    stack = _stack(source_commit)
    inventory = _inventory(source_commit, stack)
    review = _sign(
        {
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
        },
        "license_review_receipt_digest",
    )
    envelope = _sign(
        {
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
            "license_review_receipt_digest": review["license_review_receipt_digest"],
        },
        "paid_execution_envelope_digest",
    )
    packet = prepare_reconstruction_worker_remote_build_packet(
        output_dir=tmp_path / "packet",
        repo_root=repo,
        image_ref=DEFAULT_IMAGE_REF,
        source_commit=source_commit,
        source_worktree_dirty=False,
        worker_stack_manifest=stack,
        license_inventory=inventory,
        license_review_receipt=review,
        paid_execution_envelope=envelope,
        generated_at="2026-08-01T12:01:00Z",
    )
    assert packet["status"] == "ready"
    image = DEFAULT_IMAGE_REF.rsplit(":", 1)[0] + "@sha256:" + "a" * 64
    remote = _sign(
        {
            "schema_version": "reconstruction_worker_build_receipt.v2",
            "timestamp": "2026-08-01T12:02:00Z",
            "worker_stack_manifest_digest": stack["worker_stack_manifest_digest"],
            "license_inventory_digest": inventory["license_inventory_digest"],
            "license_review_receipt_digest": review["license_review_receipt_digest"],
            "paid_execution_envelope_digest": envelope["paid_execution_envelope_digest"],
            "status": "built",
            "resolved_image_digest": image,
            "source_commit_sha": source_commit,
            "build_context_digest": "sha256:" + packet["context_manifest_sha256"],
            "duration_seconds": 100.0,
            "cost_usd": 0.0,
            "logs": ["reconstruction_worker_build_metadata.json"],
            "blockers": [],
            "scientific_qualification_inferred": False,
            "build_healthcheck_embedded": True,
            "runtime_gpu_healthcheck_completed": False,
            "raw_secret_values_recorded": False,
            "proof_effect": "none",
            "claim_ceiling": "resolved_worker_image_build_only",
        },
        "build_receipt_digest",
    )
    teardown = {
        "schema_version": "groot_oscar_digitalocean_builder_teardown.v1",
        "droplet_id": 42,
        "delete_http_status": 204,
        "verify_http_status": 404,
        "provider_absence_confirmed": True,
        "elapsed_seconds": 120.0,
        "maximum_compute_spend_usd": 0.2,
        "raw_secret_values_recorded": False,
    }
    builder = {
        "schema_version": "groot_oscar_digitalocean_builder_run.v1",
        "status": "completed",
        "blockers": [],
        "droplet_id": 42,
        "build_exit_code": 0,
        "source_commit": source_commit,
        "packet_kind": "reconstruction_worker_image",
        "local_capability_cleanup_verified": True,
        "provider_absence_confirmed": True,
        "maximum_compute_spend_usd": 0.2,
        "raw_secret_values_recorded": False,
        "claim_boundary": {
            "image_build_is_not_model_cache_verification": True,
            "image_build_is_not_runpod_startup": True,
            "image_build_is_not_task_success": True,
        },
    }
    return {
        "worker_stack_manifest": stack,
        "remote_build_packet": packet,
        "remote_build_receipt": remote,
        "builder_run_result": builder,
        "teardown_receipt": teardown,
    }


def test_normalizes_exact_bound_build_and_provider_zero_evidence(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    receipt = compile_reconstruction_worker_build_receipt(**inputs)

    assert receipt["status"] == "built"
    assert receipt["provider_zero_verified"] is True
    assert receipt["runtime_gpu_healthcheck_completed"] is False
    assert receipt["proof_effect"] == "none"
    assert receipt["cost_usd"] == 0.2
    assert receipt["build_receipt_digest"] == canonical_digest(
        receipt, digest_field="build_receipt_digest"
    )


@pytest.mark.parametrize(
    ("target", "field", "value", "expected"),
    [
        (
            "remote_build_receipt",
            "source_commit_sha",
            "f" * 40,
            "worker_build_artifact_binding_mismatch",
        ),
        (
            "builder_run_result",
            "provider_absence_confirmed",
            False,
            "worker_build_outer_builder_result_not_accepted",
        ),
        (
            "teardown_receipt",
            "provider_absence_confirmed",
            False,
            "worker_build_teardown_not_accepted",
        ),
        (
            "teardown_receipt",
            "maximum_compute_spend_usd",
            2.0,
            "worker_build_cost_or_duration_outside_envelope",
        ),
        (
            "remote_build_receipt",
            "resolved_image_digest",
            "attacker.invalid/worker@sha256:" + "b" * 64,
            "worker_build_resolved_image_binding_invalid",
        ),
    ],
)
def test_rejects_stale_teardown_or_out_of_envelope_evidence(
    tmp_path: Path, target: str, field: str, value: object, expected: str
) -> None:
    inputs = _inputs(tmp_path)
    inputs[target][field] = value
    if target == "remote_build_receipt":
        _sign(inputs[target], "build_receipt_digest")

    with pytest.raises(ReconstructionWorkerBuildNormalizationError) as caught:
        compile_reconstruction_worker_build_receipt(**inputs)
    assert expected in caught.value.codes


def test_rejects_packet_tarball_tamper(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    tarball = Path(inputs["remote_build_packet"]["tarball_path"])
    tarball.write_bytes(tarball.read_bytes() + b"tamper")

    with pytest.raises(ReconstructionWorkerBuildNormalizationError) as caught:
        compile_reconstruction_worker_build_receipt(**inputs)
    assert "worker_build_remote_packet_archive_invalid" in caught.value.codes
