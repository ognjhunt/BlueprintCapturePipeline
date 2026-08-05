from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from blueprint_pipeline import adp_inpaint360_interiorgs_vast as runtime
from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionGrant
from blueprint_pipeline.vast_provider_adapter import (
    _blueprint_bundle_preflight,
    _probe_env,
    _probe_shell_script,
)
from blueprint_pipeline.wam_provider_output import inspect_provider_runtime_output_zip


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _load_provider_runner():
    path = (
        Path(__file__).resolve().parents[1] / "scripts/adp_inpaint360_interiorgs_provider_runner.py"
    )
    spec = importlib.util.spec_from_file_location("adp_inpaint360_provider_runner_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    real_root = Path(__file__).resolve().parents[1]
    for name in (
        "run_adp_inpaint360_interiorgs_provider_runtime.sh",
        "adp_inpaint360_interiorgs_provider_runner.py",
        "materialize_inpaint360_virtual_masks.py",
    ):
        (scripts / name).write_bytes((real_root / "scripts" / name).read_bytes())
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    source = tmp_path / "Inpaint360GS"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    license_path = source / "LICENSE.txt"
    license_path.write_text("publisher license", encoding="utf-8")
    (source / "README.md").write_text("publisher source", encoding="utf-8")
    (source / "docs").mkdir()
    (source / "docs/target.md").write_text("target", encoding="utf-8")
    os.symlink("target.md", source / "docs/link.md")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(runtime, "SOURCE_COMMIT", commit)
    monkeypatch.setattr(runtime, "SOURCE_TREE", tree)
    monkeypatch.setattr(runtime, "SOURCE_LICENSE_SHA256", runtime._sha256(license_path))
    packet = tmp_path / "adapter"
    required = {
        "config/distill.json": b"{}\n",
        "config/object_removal/blueprint/840313.json": b"{}\n",
        "config/object_inpaint/blueprint/840313.json": b"{}\n",
        "vanilla_3dgs/cfg_args": b"Namespace()\n",
        "vanilla_3dgs/point_cloud/iteration_30000/point_cloud.ply": b"ply fixture",
        "source/images/view.png": b"image",
    }
    records = []
    for relative, content in required.items():
        path = packet / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        records.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": runtime._sha256(path),
            }
        )
    adapter_receipt: dict[str, object] = {
        "status": "prepared_unexecuted",
        "scene": {
            "publisher_scene_id": runtime.SCENE_ID,
            "target_instance_id": runtime.TARGET_INSTANCE_ID,
            "target_semantic_label": "canned_beverage",
        },
        "source": {"commit": commit, "tree": tree},
        "adapter": {
            "target_method_instance_id": runtime.TARGET_METHOD_INSTANCE_ID,
            "target_object_radius_m": 0.095,
            "target_object_radius_derivation": "max_distance_from_metric_obb_center",
            "staged_artifacts": records,
        },
    }
    adapter_receipt["receipt_digest"] = canonical_digest(
        adapter_receipt, digest_field="receipt_digest"
    )
    adapter_receipt_path = packet / "receipt.json"
    _write_json(adapter_receipt_path, adapter_receipt)
    big_lama = tmp_path / "big-lama.zip"
    big_lama.write_bytes(b"big lama fixture")
    monkeypatch.setattr(runtime, "BIG_LAMA_SIZE_BYTES", big_lama.stat().st_size)
    monkeypatch.setattr(runtime, "BIG_LAMA_SHA256", runtime._sha256(big_lama))
    prerequisite: dict[str, object] = {
        "methods": {
            "inpaint360_author_smoke": {
                "artifacts": [
                    {
                        "artifact_id": "big_lama_author_linked_archive",
                        "rights_established": True,
                        "rights_authority_id": "big_lama_apache_2_0",
                        "size_bytes": big_lama.stat().st_size,
                        "sha256": runtime._sha256(big_lama),
                    }
                ]
            }
        }
    }
    prerequisite["receipt_digest"] = canonical_digest(prerequisite, digest_field="receipt_digest")
    monkeypatch.setattr(runtime, "PREREQUISITE_RECEIPT_DIGEST", prerequisite["receipt_digest"])
    prerequisite_path = tmp_path / "prerequisite.json"
    _write_json(prerequisite_path, prerequisite)
    return {
        "repo": repo,
        "source": source,
        "packet": packet,
        "adapter_receipt": adapter_receipt_path,
        "prerequisite": prerequisite_path,
        "big_lama": big_lama,
        "job": tmp_path / "job",
    }


def _build(paths: dict[str, Path]) -> dict[str, object]:
    return runtime.build_inpaint360_interiorgs_bundle(
        repo_root=paths["repo"],
        inpaint360_root=paths["source"],
        adapter_root=paths["packet"],
        adapter_receipt_path=paths["adapter_receipt"],
        prerequisite_receipt_path=paths["prerequisite"],
        big_lama_path=paths["big_lama"],
        job_dir=paths["job"],
        generated_at="2026-08-05T00:00:00+00:00",
    )


def test_bundle_binds_source_packet_rights_and_two_environments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = _build(paths)
    assert receipt["status"] == "ready"
    assert receipt["retry_cap"] == 0
    assert receipt["adapter_receipt_digest"]
    runtime_root = paths["job"] / "provider_runtime"
    spec = json.loads((runtime_root / "execution_spec.json").read_text())
    assert spec["runtime"]["main_torch"] == "2.0.0+cu118"
    assert spec["runtime"]["lama_torch"] == "1.8.0+cu111"
    assert spec["runtime"]["lama_environment_relation"] == (
        "compatibility_environment_not_exact_publisher_conda_env"
    )
    assert spec["runtime"]["source_modifications"] == []
    with tarfile.open(runtime_root / "inpaint360gs_source.tar") as archive:
        link = archive.getmember("docs/link.md")
        assert link.issym()
        assert link.linkname == "target.md"
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
    assert "provider_runtime/execution_spec.json" in names
    assert "provider_runtime/big-lama.zip" in names


def test_bundle_passes_vast_preflight_and_has_fail_closed_launch_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = _build(paths)
    env = _probe_env(
        job_dir=tmp_path,
        enable_isaac_smoke=False,
        provider_bundle_kind=runtime.PROVIDER_BUNDLE_KIND,
        forward_hf_token=False,
    )
    assert env["NVIDIA_DRIVER_CAPABILITIES"] == "all"
    assert "ACCEPT_EULA" not in env
    script = _probe_shell_script(
        "https://example.test",
        enable_isaac_smoke=False,
        enable_blueprint_bundle=True,
        provider_bundle_kind=runtime.PROVIDER_BUNDLE_KIND,
    )
    assert "run_adp_inpaint360_interiorgs_provider_runtime.sh" in script
    assert 'curl --http1.1 -fL "$blueprint_download_src"' in script
    assert "adp_inpaint360_provider_runtime_output.zip" in script
    assert "provider_output_zip_exclusions.json" in script
    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="fixed",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind=runtime.PROVIDER_BUNDLE_KIND,
        bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://example.test/bundle.zip?signature=redacted",
        provider_output_put_url="https://example.test/output.zip?signature=redacted",
    )
    assert preflight["status"] == "passed"
    assert preflight["blockers"] == []


def test_bundle_rejects_mutated_adapter_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    (paths["packet"] / "config/distill.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="adapter_artifact_changed"):
        _build(paths)


def test_bundle_rejects_dirty_blueprint_runtime_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    (paths["repo"] / "scripts/adp_inpaint360_interiorgs_provider_runner.py").write_text(
        "changed", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="blueprint_repository_tracked_state_dirty"):
        _build(paths)


def test_bundle_rejects_caller_asserted_big_lama_rights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    prerequisite = json.loads(paths["prerequisite"].read_text())
    prerequisite["methods"]["inpaint360_author_smoke"]["artifacts"][0]["rights_established"] = False
    prerequisite["receipt_digest"] = canonical_digest(prerequisite, digest_field="receipt_digest")
    runtime.PREREQUISITE_RECEIPT_DIGEST = prerequisite["receipt_digest"]
    _write_json(paths["prerequisite"], prerequisite)
    with pytest.raises(ValueError, match="rights_or_identity_missing"):
        _build(paths)


def test_provider_runner_validates_target_identity_and_retains_hd_review_frames(
    tmp_path: Path,
) -> None:
    runner = _load_provider_runner()
    source = tmp_path / "source"
    raw = source / "raw_hqsam"
    associated = source / "associated_hqsam"
    raw.mkdir(parents=True)
    associated.mkdir(parents=True)
    for index in range(2):
        Image.new("L", (4, 3), color=1).save(raw / f"view_{index}.png")
        Image.new("L", (4, 3), color=1).save(associated / f"view_{index}.png")
    _write_json(associated / "scene.json", {"num_classes": 2})
    output = tmp_path / "output"
    receipt = runner._validate_mask_association(
        source_data=source,
        target_method_instance_id=1,
        output=output,
    )
    assert receipt["status"] == "accepted"
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    for index in range(10):
        Image.new("RGB", (16, 6), color=(index, 0, 0)).save(render_dir / f"{index:05d}.png")
    frames = runner._retain_review_frames(render_dir, output)
    assert len(frames) == 8
    assert frames[0]["rgb"]["width"] == 8
    assert frames[0]["rgb"]["height"] == 6
    assert frames[0]["rgb_and_mask"]["width"] == 16


def _allocator_args(tmp_path: Path, receipt: Path, *, execute: bool) -> list[str]:
    args = [
        "gpu-canary",
        "--probe-kind",
        runtime.PROBE_KIND,
        "--provider",
        "vast",
        "--provider-launch-request",
        str(tmp_path / "unused-request.json"),
        "--release-evidence",
        str(tmp_path / "unused-release.json"),
        "--model-cache-evidence",
        str(tmp_path / "unused-model.json"),
        "--preflight-bundle",
        str(tmp_path / "unused-preflight.json"),
        "--admission-out",
        str(tmp_path / "admission.json"),
        "--bound-request-out",
        str(tmp_path / "unused-bound.json"),
        "--adapter-output",
        str(tmp_path / "adapter.json"),
        "--pod-name",
        "adp-inpaint360-interiorgs",
        "--adp-inpaint360-bundle-receipt",
        str(receipt),
        "--adp-job-dir",
        str(tmp_path / "run"),
        "--adp-max-hourly-rate-usd",
        "1.50",
        "--adp-max-spend-usd",
        "6.00",
        "--adp-hard-ttl-seconds",
        "14400",
    ]
    if execute:
        args.append("--execute")
    return args


@pytest.mark.parametrize("execute", [False, True])
def test_canonical_allocator_issues_inpaint360_grant_only_for_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execute: bool
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = _build(paths)
    receipt_path = paths["job"] / "adp_inpaint360_interiorgs_bundle_receipt.json"
    monkeypatch.setattr(
        allocator,
        "ADP_INPAINT360_SOURCE_COMMIT",
        receipt["source_commit"],
    )
    monkeypatch.setattr(
        allocator,
        "ADP_INPAINT360_SOURCE_TREE",
        receipt["source_tree"],
    )
    monkeypatch.setattr(
        allocator,
        "ADP_INPAINT360_PREREQUISITE_RECEIPT_DIGEST",
        receipt["prerequisite_receipt_digest"],
    )
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    observed: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, str]:
        observed.update(kwargs)
        return {"status": "completed" if kwargs["execute"] else "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_inpaint360_interiorgs_vast", fake_run)
    assert allocator.main(_allocator_args(tmp_path, receipt_path, execute=execute)) == 0
    assert observed["execute"] is execute
    assert (
        isinstance(observed["paid_resource_admission_grant"], PaidResourceAdmissionGrant) is execute
    )
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["retry_cap"] == 0
    assert admission["rendered_frames_have_no_hidden_background_truth"] is True
    assert admission["replacement_or_physics_result_claimed"] is False
    assert admission["hard_cap_usd"] == 6.0


def test_output_inspector_recognizes_inpaint360_runtime_result(tmp_path: Path) -> None:
    output = tmp_path / "output.zip"
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            "adp_inpaint360_interiorgs_result.json",
            json.dumps({"status": "blocked", "blockers": ["typed_runtime_failure"]}),
        )
    inspection = inspect_provider_runtime_output_zip(output)
    assert inspection["runtime_result_present"] is True
    assert inspection["runtime_result_blockers"] == ["typed_runtime_failure"]


def test_live_runner_dry_run_performs_no_provider_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = _build(paths)
    result = runtime.run_inpaint360_interiorgs_vast(
        job_dir=tmp_path / "run",
        paid_resource_admission_grant=None,
        execute=False,
        prepared_bundle=receipt,
        public_image=runtime.DEFAULT_IMAGE,
    )
    assert result["status"] == "dry_run_ready"
    assert result["provider_mutations_performed"] == 0
    assert result["retry_cap"] == 0


def test_fresh_live_job_has_full_budget_without_preexisting_ledger(tmp_path: Path) -> None:
    assert runtime._remaining_minutes(
        job=tmp_path / "fresh-job",
        hard_cap_usd=6.0,
        hard_ttl_seconds=14_400,
        max_hourly_rate_usd=1.5,
    ) == 240
