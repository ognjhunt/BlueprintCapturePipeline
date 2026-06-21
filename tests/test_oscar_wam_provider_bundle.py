from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import oscar_wam_provider_bundle as bundle_module
from blueprint_pipeline.oscar_wam_provider_bundle import build_oscar_wam_provider_bundle


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_build_oscar_wam_provider_bundle_from_existing_inputs(tmp_path: Path) -> None:
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    rollout_input.write_text(
        json.dumps(
            {
                "schema_version": "wam_rollout_input_manifest.v1",
                "source_mujoco_endpoint_eval_job_dir": str(tmp_path / "mujoco-job"),
                "task_prompts": [{"task_prompt": "Reach toward the object."}],
            }
        ),
        encoding="utf-8",
    )
    oscar_input = tmp_path / "oscar_input"
    oscar_input.mkdir()
    (oscar_input / "first_frame.png").write_bytes(b"fake-png")
    (oscar_input / "blueprint_proxy_skeleton_conditioning.mp4").write_bytes(b"fake-mp4")

    manifest = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "bundle-job",
        wam_rollout_input_manifest=rollout_input,
        oscar_input_dir=oscar_input,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert manifest["status"] == "completed"
    bundle_path = Path(str(manifest["bundle_path"]))
    assert bundle_path.is_file()
    persisted = _read_json(tmp_path / "bundle-job" / "oscar_wam_provider_bundle_manifest.json")
    assert persisted["raw_credentials_written_to_artifacts"] is False
    with zipfile.ZipFile(bundle_path) as archive:
        names = set(archive.namelist())
    assert "provider_runtime/run_wam_provider_runtime.sh" in names
    assert "provider_runtime/wam_provider_runtime_runner.py" in names
    assert "provider_runtime/wam_provider_runtime_manifest.json" in names
    assert "provider_runtime/wam_rollout_input_manifest.json" in names
    assert "provider_runtime/oscar_input/first_frame.png" in names
    assert "provider_runtime/oscar_input/blueprint_proxy_skeleton_conditioning.mp4" in names
    with zipfile.ZipFile(bundle_path) as archive:
        runner_text = archive.read("provider_runtime/wam_provider_runtime_runner.py").decode(
            "utf-8"
        )
    assert 'checkpoint_path / "model"' in runner_text
    assert "inference_checkpoint_path" in runner_text


def test_build_oscar_wam_provider_bundle_blocks_missing_rollout_input(tmp_path: Path) -> None:
    manifest = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "bundle-job",
        wam_rollout_input_manifest=tmp_path / "missing.json",
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert manifest["status"] == "blocked"
    assert "wam_rollout_input_manifest_missing" in manifest["blockers"]


def test_oscar_wam_provider_bundle_existing_input_edges(tmp_path: Path) -> None:
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    rollout_input.write_text(
        json.dumps({"source_mujoco_endpoint_eval_job_dir": str(tmp_path / "mujoco-job")}),
        encoding="utf-8",
    )
    oscar_input = tmp_path / "oscar_input"
    oscar_input.mkdir()
    with pytest.raises(FileNotFoundError, match="oscar_input_first_frame_missing"):
        bundle_module._materialized_package_from_existing(
            oscar_input_dir=oscar_input,
            package_manifest_path=None,
            rollout_manifest={},
        )

    (oscar_input / "first_frame.png").write_bytes(b"png")
    with pytest.raises(
        FileNotFoundError,
        match="oscar_input_skeleton_conditioning_video_missing",
    ):
        bundle_module._materialized_package_from_existing(
            oscar_input_dir=oscar_input,
            package_manifest_path=None,
            rollout_manifest={},
        )

    (oscar_input / "blueprint_proxy_skeleton_conditioning.mp4").write_bytes(b"mp4")
    package_manifest = tmp_path / "package.json"
    package_manifest.write_text(
        json.dumps(
            {
                "first_frame": {"path": "old.png", "width": 64},
                "skeleton_video": {"path": "old.mp4", "fps": 5},
                "claim_boundary": {"existing": True},
            }
        ),
        encoding="utf-8",
    )
    package = bundle_module._materialized_package_from_existing(
        oscar_input_dir=oscar_input,
        package_manifest_path=package_manifest,
        rollout_manifest={},
    )
    assert package["first_frame"]["path"] == str(oscar_input / "first_frame.png")
    assert package["first_frame"]["width"] == 64
    assert package["claim_boundary"]["existing"] is True


def test_oscar_wam_provider_bundle_local_materialization_and_failure_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    rollout_input.write_text(
        json.dumps(
            {
                "schema_version": "wam_rollout_input_manifest.v1",
                "source_mujoco_endpoint_eval_job_dir": str(tmp_path / "mujoco-job"),
            }
        ),
        encoding="utf-8",
    )
    job_dir = tmp_path / "bundle-job"
    (job_dir / "oscar_wam_provider_bundle").mkdir(parents=True)
    captured: dict[str, Any] = {}

    def fake_materialize(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        first = kwargs["work_dir"] / "first_frame.png"
        skeleton = kwargs["work_dir"] / "blueprint_proxy_skeleton_conditioning.mp4"
        first.parent.mkdir(parents=True, exist_ok=True)
        first.write_bytes(b"png")
        skeleton.write_bytes(b"mp4")
        return {
            "schema_version": "blueprint_oscar_wam_input_package.v1",
            "first_frame": {"path": str(first)},
            "skeleton_video": {"path": str(skeleton)},
            "claim_boundary": {},
        }

    monkeypatch.setattr(bundle_module, "_materialize_oscar_input_package", fake_materialize)
    manifest = build_oscar_wam_provider_bundle(
        job_dir=job_dir,
        wam_rollout_input_manifest=rollout_input,
        generated_at="2026-06-21T00:00:00+00:00",
    )
    assert manifest["status"] == "completed"
    assert captured["width"] == bundle_module.DEFAULT_WIDTH
    assert not (job_dir / "oscar_wam_provider_bundle" / "stale").exists()

    def raise_materialize(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise RuntimeError("cannot render inputs")

    monkeypatch.setattr(bundle_module, "_materialize_oscar_input_package", raise_materialize)
    blocked = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "blocked-job",
        wam_rollout_input_manifest=rollout_input,
        generated_at="2026-06-21T00:00:00+00:00",
    )
    assert blocked["status"] == "blocked"
    assert blocked["blockers"] == ["oscar_wam_input_package_materialization_failed:RuntimeError"]


def test_oscar_wam_provider_bundle_zip_integrity_and_cli_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    rollout_input.write_text(json.dumps({"task_prompts": []}), encoding="utf-8")
    oscar_input = tmp_path / "oscar_input"
    oscar_input.mkdir()
    (oscar_input / "first_frame.png").write_bytes(b"png")
    (oscar_input / "blueprint_proxy_skeleton_conditioning.mp4").write_bytes(b"mp4")
    real_zipfile = bundle_module.zipfile.ZipFile

    class CorruptReadZip:
        def __enter__(self) -> "CorruptReadZip":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def namelist(self) -> list[str]:
            return ["provider_runtime/run_wam_provider_runtime.sh"]

        def testzip(self) -> str:
            return "provider_runtime/run_wam_provider_runtime.sh"

    def zip_factory(path: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if mode == "w":
            return real_zipfile(path, mode, *args, **kwargs)
        return CorruptReadZip()

    monkeypatch.setattr(bundle_module.zipfile, "ZipFile", zip_factory)
    corrupt = build_oscar_wam_provider_bundle(
        job_dir=tmp_path / "corrupt-job",
        wam_rollout_input_manifest=rollout_input,
        oscar_input_dir=oscar_input,
        generated_at="2026-06-21T00:00:00+00:00",
    )
    assert corrupt["status"] == "blocked"
    assert corrupt["blockers"] == ["provider_runtime_bundle_zip_integrity_failed"]

    captured: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "blocked", "blockers": ["cli_blocker"]}

    monkeypatch.setattr(bundle_module, "build_oscar_wam_provider_bundle", fake_build)
    code = bundle_module.main(
        [
            "--job-dir",
            str(tmp_path / "cli-job"),
            "--wam-rollout-input-manifest",
            str(rollout_input),
            "--oscar-input-dir",
            str(oscar_input),
            "--oscar-input-package-manifest",
            str(tmp_path / "package.json"),
            "--oscar-source-url",
            "https://example.com/oscar.git",
            "--oscar-hf-repo",
            "example/oscar",
            "--timeout-seconds",
            "12",
            "--num-steps",
            "7",
            "--guidance",
            "3.5",
            "--seed",
            "99",
            "--bundle-filename",
            "custom.zip",
        ]
    )
    output = capsys.readouterr().out
    assert code == 1
    assert "[oscar-wam-provider-bundle] blockers=cli_blocker" in output
    assert captured["oscar_source_url"] == "https://example.com/oscar.git"
    assert captured["oscar_hf_repo"] == "example/oscar"
    assert captured["timeout_seconds"] == 12
    assert captured["num_steps"] == 7
    assert captured["guidance"] == 3.5
    assert captured["seed"] == 99
    assert captured["bundle_filename"] == "custom.zip"
