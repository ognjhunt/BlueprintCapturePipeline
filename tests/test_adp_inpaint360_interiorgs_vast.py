from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
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


def _load_rasterizer_probe():
    path = Path(__file__).resolve().parents[1] / "scripts/probe_inpaint360_camera_rasterizer.py"
    spec = importlib.util.spec_from_file_location("inpaint360_camera_rasterizer_test", path)
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
        "probe_inpaint360_camera_rasterizer.py",
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
    lama_source = tmp_path / "LaMa-upstream"
    lama_source.mkdir()
    subprocess.run(["git", "init", "-q", str(lama_source)], check=True)
    subprocess.run(
        ["git", "-C", str(lama_source), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(lama_source), "config", "user.name", "Test"], check=True
    )
    lama_license = lama_source / "LICENSE"
    lama_license.write_text("Apache fixture", encoding="utf-8")
    for relative in runtime.LAMA_REQUIRED_RUNTIME_FILES:
        path = lama_source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(lama_source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(lama_source), "commit", "-qm", "fixture"], check=True)
    lama_commit = subprocess.run(
        ["git", "-C", str(lama_source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lama_tree = subprocess.run(
        ["git", "-C", str(lama_source), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(runtime, "LAMA_SOURCE_COMMIT", lama_commit)
    monkeypatch.setattr(runtime, "LAMA_SOURCE_TREE", lama_tree)
    monkeypatch.setattr(
        runtime, "LAMA_SOURCE_LICENSE_SHA256", runtime._sha256(lama_license)
    )
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
            "target_obb_corners_m": [
                [float(x), float(y), float(z)]
                for x in (0, 1)
                for y in (0, 1)
                for z in (0, 1)
            ],
            "target_removal_volume_contract": (
                "gaussian_center_inside_exact_publisher_obb"
            ),
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
    vgg16_weights = tmp_path / runtime.VGG16_WEIGHTS_FILENAME
    vgg16_weights.write_bytes(b"vgg16 fixture")
    monkeypatch.setattr(runtime, "VGG16_WEIGHTS_SIZE_BYTES", vgg16_weights.stat().st_size)
    monkeypatch.setattr(runtime, "VGG16_WEIGHTS_SHA256", runtime._sha256(vgg16_weights))
    return {
        "repo": repo,
        "source": source,
        "lama_source": lama_source,
        "packet": packet,
        "adapter_receipt": adapter_receipt_path,
        "prerequisite": prerequisite_path,
        "big_lama": big_lama,
        "vgg16_weights": vgg16_weights,
        "job": tmp_path / "job",
    }


def _build(paths: dict[str, Path]) -> dict[str, object]:
    return runtime.build_inpaint360_interiorgs_bundle(
        repo_root=paths["repo"],
        inpaint360_root=paths["source"],
        lama_root=paths["lama_source"],
        adapter_root=paths["packet"],
        adapter_receipt_path=paths["adapter_receipt"],
        prerequisite_receipt_path=paths["prerequisite"],
        big_lama_path=paths["big_lama"],
        vgg16_weights_path=paths["vgg16_weights"],
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
    assert spec["runtime"]["mask_association_mode"] == (
        "pre_registered_single_target_resolution_divisor_2"
    )
    assert spec["runtime"]["method_resolution_argument"] == 2
    assert spec["runtime"]["method_input_resolution"] == "1024x768"
    adapter_payload = json.loads(paths["adapter_receipt"].read_text())
    assert spec["target_obb_corners_m"] == adapter_payload["adapter"][
        "target_obb_corners_m"
    ]
    assert spec["target_removal_volume_contract"] == (
        "gaussian_center_inside_exact_publisher_obb"
    )
    assert spec["nested_dependencies"]["lama"]["commit"]
    assert spec["nested_dependencies"]["lama"]["tree"]
    assert spec["nested_dependencies"]["lama"]["materialization"] == (
        "add_missing_publisher_runtime_modules_without_overwrite"
    )
    assert spec["runtime"]["source_modifications"] == []
    assert spec["runtime"]["source_materialization_additions"] == list(
        runtime.LAMA_REQUIRED_RUNTIME_FILES
    )
    entrypoint = (
        runtime_root / "run_adp_inpaint360_interiorgs_provider_runtime.sh"
    ).read_text(encoding="utf-8")
    assert 'pip uninstall --python "${LAMA_PY}" opencv-python' in entrypoint
    assert 'pip install --python "${LAMA_PY}" --reinstall --no-deps \\' in entrypoint
    assert "opencv-python-headless==4.5.5.64" in entrypoint
    assert "inpaint360_lama_opencv_conflict_resolution_failed" in entrypoint
    assert "inpaint360_lama_opencv_runtime_validation_failed" in entrypoint
    assert 'metadata.version("opencv-python-headless") == "4.5.5.64"' in entrypoint
    assert 'numpy.__version__ == "1.21.6"' in entrypoint
    with tarfile.open(runtime_root / "inpaint360gs_source.tar") as archive:
        link = archive.getmember("docs/link.md")
        assert link.issym()
        assert link.linkname == "target.md"
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
    assert "provider_runtime/execution_spec.json" in names
    assert "provider_runtime/big-lama.zip" in names
    assert f"provider_runtime/{runtime.VGG16_WEIGHTS_FILENAME}" in names
    assert "provider_runtime/lama_training_data.zip" in names
    assert "provider_runtime/probe_inpaint360_camera_rasterizer.py" in names


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
    assert "BLUEPRINT_ADP_INPAINT360_RUNTIME_PROGRESS" in (
        Path(__file__).resolve().parents[1]
        / "scripts/run_adp_inpaint360_interiorgs_provider_runtime.sh"
    ).read_text(encoding="utf-8")
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


def test_bundle_rejects_mutated_lama_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    (paths["lama_source"] / runtime.LAMA_REQUIRED_RUNTIME_FILES[0]).write_text(
        "changed", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="lama_dependency_identity_mismatch"):
        _build(paths)


def test_bundle_rejects_mutated_vgg16_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    paths["vgg16_weights"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="vgg16_weights_bytes_changed"):
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


def test_provider_runner_retains_hd_review_frames(
    tmp_path: Path,
) -> None:
    runner = _load_provider_runner()
    output = tmp_path / "output"
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    for index in range(10):
        Image.new("RGB", (16, 6), color=(index, 0, 0)).save(render_dir / f"{index:05d}.png")
    frames = runner._retain_review_frames(render_dir, output)
    assert len(frames) == 8
    assert frames[0]["rgb"]["width"] == 8
    assert frames[0]["rgb"]["height"] == 6
    assert frames[0]["rgb_and_mask"]["width"] == 16


def test_provider_runner_prefers_method_local_tools_over_installed_package(
    tmp_path: Path,
) -> None:
    runner = _load_provider_runner()
    source = tmp_path / "Inpaint360GS"
    script_dir = source / "seg"
    method_tools = source / "tools"
    installed = tmp_path / "site-packages"
    installed_tools = installed / "tools"
    script_dir.mkdir(parents=True)
    method_tools.mkdir()
    installed_tools.mkdir(parents=True)
    (method_tools / "__init__.py").write_text("IDENTITY = 'method'\n", encoding="utf-8")
    (installed_tools / "__init__.py").write_text("IDENTITY = 'installed'\n", encoding="utf-8")

    env = runner._prepend_pythonpath({"PYTHONPATH": str(installed)}, source)
    completed = subprocess.run(
        [sys.executable, "-c", "import tools; print(tools.IDENTITY)"],
        cwd=script_dir,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "method"
    assert env["PYTHONPATH"].split(os.pathsep) == [str(source.resolve()), str(installed)]


def test_provider_runner_materializes_pre_registered_single_target_masks(
    tmp_path: Path,
) -> None:
    runner = _load_provider_runner()
    source = tmp_path / "source"
    raw = source / "raw_hqsam"
    images = source / "images"
    raw.mkdir(parents=True)
    images.mkdir(parents=True)
    for index in range(3):
        image = Image.new("L", (10, 8), color=0)
        image.paste(1, (index * 2, index * 2, index * 2 + 2, index * 2 + 2))
        image.save(raw / f"view_{index}.png")
        Image.new("RGB", (10, 8), color=(index, 0, 0)).save(
            images / f"view_{index}.png"
        )

    receipt = runner._materialize_pre_registered_mask_binding(
        source_data=source,
        target_method_instance_id=1,
        output=tmp_path / "output",
    )

    assert receipt["status"] == "accepted"
    assert receipt["full_resolution_source_preserved"] is True
    assert receipt["image_mask_dimensions"]["view_0.png"] == {
        "source_image": [10, 8],
        "raw_mask": [10, 8],
        "method_image_and_associated_mask": [5, 4],
    }
    assert receipt["target_pixel_counts"] == {
        "view_0.png": 4,
        "view_1.png": 4,
        "view_2.png": 4,
    }
    associated = source / "associated_hqsam"
    for path in sorted(raw.glob("*.png")):
        with Image.open(associated / path.name) as derived:
            assert derived.size == (5, 4)
            assert set(derived.getdata()) == {0, 1}
        assert (associated / path.name).read_bytes() != path.read_bytes()
    scene = json.loads((associated / "scene.json").read_text())
    assert scene["num_classes"] == 2
    assert scene["association_mode"] == (
        "pre_registered_single_target_resolution_divisor_2"
    )


@pytest.mark.parametrize("invalid_kind", ["empty_target", "unexpected_instance"])
def test_provider_runner_rejects_invalid_pre_registered_target_masks(
    tmp_path: Path, invalid_kind: str
) -> None:
    runner = _load_provider_runner()
    source = tmp_path / "source"
    raw = source / "raw_hqsam"
    images = source / "images"
    raw.mkdir(parents=True)
    images.mkdir(parents=True)
    value = 0 if invalid_kind == "empty_target" else 2
    Image.new("L", (5, 4), color=value).save(raw / "bad.png")
    Image.new("RGB", (5, 4)).save(images / "bad.png")

    receipt = runner._materialize_pre_registered_mask_binding(
        source_data=source,
        target_method_instance_id=1,
        output=tmp_path / "output",
    )

    assert receipt["status"] == "blocked"
    assert receipt["invalid_masks"] == ["bad.png"]
    assert not (source / "associated_hqsam").exists()


@pytest.mark.parametrize("invalid_kind", ["missing_image", "dimension_mismatch"])
def test_provider_runner_rejects_unpaired_or_resized_target_masks(
    tmp_path: Path, invalid_kind: str
) -> None:
    runner = _load_provider_runner()
    source = tmp_path / "source"
    raw = source / "raw_hqsam"
    images = source / "images"
    raw.mkdir(parents=True)
    images.mkdir(parents=True)
    Image.new("L", (2048, 1536), color=1).save(raw / "view.png")
    if invalid_kind == "dimension_mismatch":
        Image.new("RGB", (1600, 1200)).save(images / "view.png")

    receipt = runner._materialize_pre_registered_mask_binding(
        source_data=source,
        target_method_instance_id=1,
        output=tmp_path / "output",
    )

    assert receipt["status"] == "blocked"
    assert receipt["full_resolution_source_preserved"] is False
    assert receipt["invalid_masks"] == ["view.png"]
    assert not (source / "associated_hqsam").exists()


def test_provider_runner_requires_method_resolution_on_image_loading_stages(
    tmp_path: Path,
) -> None:
    runner = _load_provider_runner()
    env: dict[str, str] = {}
    stages = (
        "distillation",
        "baseline_render",
        "removal",
        "virtual_views",
        "lama_prepare",
        "lama_collect",
        "ply_fusion",
        "inpaint_3d",
    )
    commands = []
    for stage in stages:
        resolution = "1" if stage in {"lama_prepare", "lama_collect"} else "2"
        commands.append(
            (stage, ["python", "script.py", "--resolution", resolution], tmp_path, env)
        )

    accepted = runner._validate_method_resolution_commands(
        commands, output=tmp_path
    )
    assert accepted["status"] == "accepted"
    commands[0][1][-1] = "-1"
    blocked = runner._validate_method_resolution_commands(
        commands, output=tmp_path
    )
    assert blocked["status"] == "blocked"
    assert blocked["violating_or_missing_stages"] == ["distillation"]


def test_provider_runner_requires_one_baseline_depth_per_frozen_camera(
    tmp_path: Path,
) -> None:
    runner = _load_provider_runner()
    source = tmp_path / "source"
    model = tmp_path / "model"
    images = source / "images"
    train_depth = model / "train/ours_2000/depth"
    test_depth = model / "test/ours_2000/depth"
    images.mkdir(parents=True)
    train_depth.mkdir(parents=True)
    test_depth.mkdir(parents=True)
    for name in ("view_a", "view_b"):
        (images / f"{name}.png").write_bytes(b"image")
    (train_depth / "view_a.npy").write_bytes(b"depth-a")
    (test_depth / "view_b.npy").write_bytes(b"depth-b")

    accepted = runner._validate_baseline_depth_inventory(
        source_data=source, model=model, iteration=2000, output=tmp_path / "output"
    )
    assert accepted["status"] == "accepted"
    assert accepted["observed_camera_splits"] == {
        "view_a": ["train"],
        "view_b": ["test"],
    }

    (test_depth / "view_b.npy").unlink()
    blocked = runner._validate_baseline_depth_inventory(
        source_data=source, model=model, iteration=2000, output=tmp_path / "output"
    )
    assert blocked["status"] == "blocked"
    assert blocked["missing_camera_names"] == ["view_b"]
    assert blocked["blockers"] == ["inpaint360_baseline_depth_inventory_invalid"]


def test_provider_runner_runs_author_baseline_render_before_removal() -> None:
    runner_path = (
        Path(__file__).resolve().parents[1]
        / "scripts/adp_inpaint360_interiorgs_provider_runner.py"
    )
    source = runner_path.read_text(encoding="utf-8")
    assert source.index('"baseline_render"') < source.index('"removal"')
    assert source.count('"render.py"') == 1
    baseline_command = source[
        source.index('(\n            "baseline_render",') : source.index(
            '(\n            "removal",'
        )
    ]
    assert '"--iteration"' not in baseline_command


def test_provider_runner_derives_divisor_2_mask_without_mutating_source(
    tmp_path: Path,
) -> None:
    runner = _load_provider_runner()
    source = tmp_path / "source"
    raw = source / "raw_hqsam"
    images = source / "images"
    raw.mkdir(parents=True)
    images.mkdir(parents=True)
    mask = Image.new("L", (2048, 1536), color=0)
    mask.paste(1, (512, 384, 1536, 1152))
    mask.save(raw / "view.png")
    Image.new("RGB", (2048, 1536), color=(20, 30, 40)).save(images / "view.png")
    raw_before = (raw / "view.png").read_bytes()

    receipt = runner._materialize_pre_registered_mask_binding(
        source_data=source,
        target_method_instance_id=1,
        output=tmp_path / "output",
    )

    assert receipt["status"] == "accepted"
    assert receipt["image_mask_dimensions"]["view.png"] == {
        "source_image": [2048, 1536],
        "raw_mask": [2048, 1536],
        "method_image_and_associated_mask": [1024, 768],
    }
    assert receipt["associated_target_pixel_counts"]["view.png"] > 0
    assert (raw / "view.png").read_bytes() == raw_before
    with Image.open(source / "associated_hqsam/view.png") as associated:
        assert associated.size == (1024, 768)
        assert set(associated.getdata()) == {0, 1}


def test_camera_rasterizer_preflight_requires_every_frozen_view() -> None:
    probe = _load_rasterizer_probe()
    rendered = {
        "image_name": "view_a.png",
        "status": "rendered",
        "maximum_projected_radius_px": 120.0,
    }
    accepted = probe._summarize([rendered, {**rendered, "image_name": "view_b.png"}], expected_count=2)
    assert accepted["status"] == "accepted"
    failed = probe._summarize(
        [rendered, {"image_name": "view_b.png", "status": "blocked"}],
        expected_count=2,
    )
    assert failed["status"] == "blocked"
    assert failed["failed_camera_names"] == ["view_b.png"]
    missing = probe._summarize([rendered], expected_count=2)
    assert "inpaint360_camera_rasterizer_view_count_mismatch" in missing["blockers"]


def test_provider_runner_enables_distillation_mode_for_camera_probe() -> None:
    runner_path = (
        Path(__file__).resolve().parents[1]
        / "scripts/adp_inpaint360_interiorgs_provider_runner.py"
    )
    source = runner_path.read_text(encoding="utf-8")
    assert source.count('"--train_distill"') == 1


def test_supplemental_fusion_view_is_frozen_from_pre_inpainting_mask_coverage(
    tmp_path: Path,
) -> None:
    runner = _load_provider_runner()
    selection = {
        "status": "accepted",
        "selected_views": [
            {"view_id": "00001", "foreground_pixels": 430, "mask_sha256": "b"},
            {"view_id": "00000", "foreground_pixels": 430, "mask_sha256": "a"},
        ],
    }

    receipt = runner._freeze_supplemental_fusion_view(
        selection=selection, output=tmp_path
    )

    assert receipt["status"] == "accepted"
    assert receipt["selected_view"] == {
        "view_id": "00000",
        "foreground_pixels": 430,
        "mask_sha256": "a",
    }
    assert receipt["selection_timing"] == "before_lama_color_depth_inpainting"


def test_nonempty_virtual_views_are_frozen_before_inpainting_and_filter_adapter_is_bound(
    tmp_path: Path,
) -> None:
    runner = _load_provider_runner()
    handoff = {
        "output_masks": [
            {
                "relative_path": "masks/00000.png",
                "foreground_pixels": 430,
                "foreground_bbox_width": 20,
                "foreground_bbox_height": 24,
                "sha256": "a",
            },
            {
                "relative_path": "masks/00001.png",
                "foreground_pixels": 0,
                "foreground_bbox_width": 0,
                "foreground_bbox_height": 0,
                "sha256": "b",
            },
            {
                "relative_path": "masks/00002.png",
                "foreground_pixels": 22,
                "foreground_bbox_width": 4,
                "foreground_bbox_height": 6,
                "sha256": "c",
            },
            {
                "relative_path": "masks/00003.png",
                "foreground_pixels": 30,
                "foreground_bbox_width": 5,
                "foreground_bbox_height": 7,
                "sha256": "d",
            },
        ]
    }
    selection = runner._freeze_nonempty_virtual_views(
        handoff=handoff,
        mask_binding={"associated_target_pixel_counts": {"view.png": 100}},
        output=tmp_path / "evidence",
    )
    source = tmp_path / "source"
    source.mkdir()
    publisher = source / "edit_object_inpaint.py"
    publisher.write_text(
        "def run():\n"
        "    virtual_pose_list = []\n"
        "    for view_tmp in []:\n"
        "        virtual_pose_list.append(view_tmp)\n\n"
        "    # 2. inpaint selected object\n"
        "    return virtual_pose_list\n",
        encoding="utf-8",
    )
    publisher_before = publisher.read_bytes()

    receipt = runner._materialize_nonempty_virtual_view_adapter(
        source=source,
        runtime=tmp_path / "runtime",
        selection=selection,
        output=tmp_path / "evidence",
    )

    assert selection["status"] == "accepted"
    assert selection["selected_count"] == 3
    assert selection["empty_view_count"] == 1
    assert selection["bbox_ineligible_view_count"] == 0
    assert selection["excluded_view_count"] == 1
    assert [row["view_id"] for row in selection["selected_views"]] == [
        "00000",
        "00002",
        "00003",
    ]
    assert receipt["status"] == "accepted"
    assert receipt["selected_view_ids"] == ["00000", "00002", "00003"]
    assert receipt["publisher_source_files_modified"] is False
    assert receipt["mask_pixels_or_images_modified"] is False
    assert receipt["unchanged_source_execution_claimed"] is False
    assert publisher.read_bytes() == publisher_before
    adapted = (tmp_path / "runtime/adapted_edit_object_inpaint.py").read_text()
    assert 'set(["00000","00002","00003"])' in adapted
    assert "if view_tmp.image_name in blueprint_nonempty_view_ids" in adapted
    compile(adapted, "adapted_edit_object_inpaint.py", "exec")


def test_nonempty_virtual_view_adapter_rejects_empty_selection_and_source_drift(
    tmp_path: Path,
) -> None:
    runner = _load_provider_runner()
    blocked_selection = runner._freeze_nonempty_virtual_views(
        handoff={
            "output_masks": [
                {
                    "relative_path": "00000.png",
                    "foreground_pixels": 0,
                    "foreground_bbox_width": 0,
                    "foreground_bbox_height": 0,
                },
                {
                    "relative_path": "00001.png",
                    "foreground_pixels": 1,
                    "foreground_bbox_width": 1,
                    "foreground_bbox_height": 1,
                }
            ]
        },
        mask_binding={"associated_target_pixel_counts": {"view.png": 100}},
        output=tmp_path / "empty",
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "edit_object_inpaint.py").write_text("publisher anchor changed\n")

    receipt = runner._materialize_nonempty_virtual_view_adapter(
        source=source,
        runtime=tmp_path / "runtime",
        selection=blocked_selection,
        output=tmp_path / "evidence",
    )

    assert blocked_selection["status"] == "blocked"
    assert blocked_selection["empty_view_count"] == 1
    assert blocked_selection["bbox_ineligible_view_count"] == 1
    assert receipt["status"] == "blocked"
    assert "inpaint360_nonempty_virtual_view_selection_not_accepted" in receipt["blockers"]
    assert "inpaint360_nonempty_virtual_view_adapter_anchor_changed" in receipt["blockers"]
    assert not (tmp_path / "runtime/adapted_edit_object_inpaint.py").exists()


def test_obb_removal_adapter_replaces_semantic_convex_hull_without_mutating_source(
    tmp_path: Path,
) -> None:
    runner = _load_provider_runner()
    source = tmp_path / "source"
    source.mkdir()
    publisher = source / "edit_object_removal.py"
    publisher.write_text(
        "import numpy as np\n"
        "import torch\n"
        "from scipy.spatial import Delaunay\n\n"
        "def removal(gaussians, mask3d):\n"
        "            mask3d_convex, object_radius = points_inside_convex_hull(gaussians._xyz.detach(), mask3d, remove_outliers=True, outlier_factor=1.0)\n"
        "           \n"
        "            mask3d = torch.logical_or(mask3d,mask3d_convex)\n"
        "            return mask3d, object_radius\n",
        encoding="utf-8",
    )
    before = publisher.read_bytes()
    spec = {
        "target_obb_corners_m": [
            [float(x), float(y), float(z)]
            for x in (0, 1)
            for y in (0, 1)
            for z in (0, 1)
        ],
        "target_removal_volume_contract": (
            "gaussian_center_inside_exact_publisher_obb"
        ),
    }

    receipt = runner._materialize_obb_removal_adapter(
        source=source,
        runtime=tmp_path / "runtime",
        spec=spec,
        output=tmp_path / "evidence",
    )

    assert receipt["status"] == "accepted"
    assert receipt["publisher_source_files_modified"] is False
    assert publisher.read_bytes() == before
    adapted = (tmp_path / "runtime/adapted_edit_object_removal.py").read_text()
    assert "blueprint_inside_obb" in adapted
    assert "torch.logical_or(mask3d,mask3d_convex)" not in adapted
    compile(adapted, "adapted_edit_object_removal.py", "exec")


def test_virtual_view_quality_gate_rejects_fragmented_and_low_coverage_masks(
    tmp_path: Path,
) -> None:
    runner = _load_provider_runner()
    selection = runner._freeze_nonempty_virtual_views(
        handoff={
            "output_masks": [
                {
                    "relative_path": "masks/00000.png",
                    "foreground_pixels": 430,
                    "foreground_bbox_width": 310,
                    "foreground_bbox_height": 303,
                    "sha256": "fragmented",
                },
                {
                    "relative_path": "masks/00001.png",
                    "foreground_pixels": 276,
                    "foreground_bbox_width": 23,
                    "foreground_bbox_height": 28,
                    "sha256": "too-small",
                },
            ]
        },
        mask_binding={"associated_target_pixel_counts": {"smallest.png": 2943}},
        output=tmp_path,
    )

    assert selection["status"] == "blocked"
    assert selection["selected_count"] == 0
    assert selection["low_fill_ratio_view_count"] == 1
    assert selection["low_reference_coverage_view_count"] == 1
    assert selection["minimum_foreground_pixels"] == 294
    assert selection["blockers"] == [
        "inpaint360_target_visible_virtual_view_support_inadequate"
    ]


def test_lama_depth_numerical_validation_rejects_non_finite_refinement(
    tmp_path: Path,
) -> None:
    runner = _load_provider_runner()
    log = tmp_path / "lama_depth.log"
    log.write_text("loss: 0.2\nloss: nan\nloss: +inf\n", encoding="utf-8")

    blocked = runner._validate_lama_depth_numerics(log_path=log, output=tmp_path)

    assert blocked["status"] == "blocked"
    assert blocked["non_finite_token_count"] == 2
    assert blocked["fusion_allowed"] is False
    assert blocked["blockers"] == ["inpaint360_lama_depth_non_finite"]

    log.write_text("loss: 0.2\nloss: 0.1\n", encoding="utf-8")
    accepted = runner._validate_lama_depth_numerics(log_path=log, output=tmp_path)
    assert accepted["status"] == "accepted"
    assert accepted["non_finite_token_count"] == 0
    assert accepted["fusion_allowed"] is True


def test_supplemental_fusion_view_materialization_rejects_empty_and_binds_selected(
    tmp_path: Path,
) -> None:
    runner = _load_provider_runner()
    fused = (
        tmp_path
        / "model/virtual/ours_object_removal/iteration_2000/fused_mask_col_dep_ply"
    )
    fused.mkdir(parents=True)
    (fused / "00000.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\nend_header\n", encoding="ascii"
    )
    (fused / "00004.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", encoding="ascii"
    )
    selection = {
        "status": "accepted",
        "selection_basis": "frozen_test",
        "selected_view": {"view_id": "00000"},
    }

    receipt = runner._materialize_supplemental_fusion_view(
        model=tmp_path / "model", selection=selection, output=tmp_path / "evidence"
    )

    assert receipt["status"] == "accepted"
    assert receipt["publisher_default_before"]["vertex_count"] == 0
    assert receipt["materialized_vertex_count"] == 2
    assert runner._ply_vertex_count(fused / "00004.ply") == 2

    empty = fused / "00002.ply"
    empty.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", encoding="ascii")
    blocked = runner._materialize_supplemental_fusion_view(
        model=tmp_path / "model",
        selection={"status": "accepted", "selected_view": {"view_id": "00002"}},
        output=tmp_path / "blocked",
    )
    assert blocked["status"] == "blocked"
    assert "inpaint360_supplemental_fusion_selected_ply_empty" in blocked["blockers"]


def _allocator_args(tmp_path: Path, receipt: Path, *, execute: bool) -> list[str]:
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
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
        "--expected-source-commit",
        receipt_payload["blueprint_repository_commit"],
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
        "ADP_INPAINT360_LAMA_SOURCE_COMMIT",
        receipt["lama_source_commit"],
    )
    monkeypatch.setattr(
        allocator,
        "ADP_INPAINT360_LAMA_SOURCE_TREE",
        receipt["lama_source_tree"],
    )
    monkeypatch.setattr(
        allocator,
        "ADP_INPAINT360_PREREQUISITE_RECEIPT_DIGEST",
        receipt["prerequisite_receipt_digest"],
    )
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: (
            [],
            {
                "orchestrator_source_commit": receipt["blueprint_repository_commit"],
                "checkout_clean": True,
            },
        ),
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
    assert admission["allocation_binding"]["expected_source_commit"] == receipt[
        "blueprint_repository_commit"
    ]


def test_canonical_allocator_rejects_mistyped_expected_source_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = _build(paths)
    receipt_path = paths["job"] / "adp_inpaint360_interiorgs_bundle_receipt.json"
    monkeypatch.setattr(allocator, "ADP_INPAINT360_SOURCE_COMMIT", receipt["source_commit"])
    monkeypatch.setattr(allocator, "ADP_INPAINT360_SOURCE_TREE", receipt["source_tree"])
    monkeypatch.setattr(
        allocator, "ADP_INPAINT360_LAMA_SOURCE_COMMIT", receipt["lama_source_commit"]
    )
    monkeypatch.setattr(
        allocator, "ADP_INPAINT360_LAMA_SOURCE_TREE", receipt["lama_source_tree"]
    )
    monkeypatch.setattr(
        allocator,
        "ADP_INPAINT360_PREREQUISITE_RECEIPT_DIGEST",
        receipt["prerequisite_receipt_digest"],
    )
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: (
            [],
            {
                "orchestrator_source_commit": receipt["blueprint_repository_commit"],
                "checkout_clean": True,
            },
        ),
    )
    args = _allocator_args(tmp_path, receipt_path, execute=False)
    args[args.index("--expected-source-commit") + 1] = "f" * 40

    assert allocator.main(args) == 2
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["status"] == "blocked"
    assert "adp_gpu_canary_expected_source_commit_not_control_plane_checkout" in admission[
        "blockers"
    ]


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


def test_live_runner_requires_observed_rasterizer_architecture_floor() -> None:
    assert runtime.MIN_RASTERIZER_COMPUTE_CAP == 890
    assert runtime.INPAINT360_GPU_SELECTION_POLICY["allowed_gpu_keywords"] == (
        "RTX 4090",
    )
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "min_compute_cap=MIN_RASTERIZER_COMPUTE_CAP" in source
    assert "gpu_selection_policy=INPAINT360_GPU_SELECTION_POLICY" in source


def test_live_runner_allows_full_scientific_ttl_with_shorter_silence_watchdog() -> None:
    assert runtime.PROVIDER_EXECUTION_TIMEOUT_SECONDS == 14_400
    assert runtime.PROVIDER_HEARTBEAT_NO_PROGRESS_SECONDS == 1800
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "startup_timeout_seconds=min(" in source
    assert "PROVIDER_EXECUTION_TIMEOUT_SECONDS, remaining_minutes * 60" in source
    assert "allowed_active_instance_ids=allowed_active_instance_ids" in source
