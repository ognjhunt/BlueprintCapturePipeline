from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from blueprint_pipeline import g1_microwave_groot_finetune_component as component
from blueprint_pipeline import g1_microwave_finetune_preflight as preflight
from blueprint_pipeline import g1_microwave_lerobot_materialization as materialization


def _dataset(root: Path) -> Path:
    for relative in component.REQUIRED_DATASET_MEMBERS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((relative + "\n").encode())
    (root / "materialization_report.json").write_text(
        json.dumps(
            {
                "schema_version": materialization.SCHEMA_VERSION,
                "status": "qualified_native_lerobot_v2_1_materialization",
                "dataset": {
                    "num_episodes": 1,
                    "num_frames": 176,
                    "fps": 50,
                    "embodiment_tag": "UNITREE_G1_SONIC",
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "groot_n17_finetune_preflight.json").write_text(
        json.dumps(
            {
                "schema_version": preflight.SCHEMA_VERSION,
                "status": "qualified_exact_groot_n1_7_training_data_preflight",
                "pinned_runtime": {
                    "groot_revision": preflight.PINNED_GROOT_N17_REVISION,
                    "warm_start_path": preflight.SEALED_SONIC_WARM_START_PATH,
                },
                "bounded_finetune_plan": {
                    "warm_starts_from_sealed_sonic_checkpoint": True,
                    "max_steps": preflight.BOUNDED_MAX_STEPS,
                    "single_gpu": True,
                    "launch_authorized": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def test_component_is_deterministic_hash_bound_and_fixed(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset")
    first = component.build_finetune_component(dataset)
    second = component.build_finetune_component(dataset)

    assert first == second
    assert first["remote_final_checkpoint"].endswith("checkpoint-500")
    assert first["dataset_archive"]["size_bytes"] > 0
    assert len(first["dataset_archive"]["sha256"]) == 64
    assert set(row["relative_path"] for row in first["dataset_archive"]["members"]) == (
        component.REQUIRED_DATASET_MEMBERS
    )
    script = first["script"]
    assert "/opt/gr00t-venv/bin/python" in script
    assert "--base-model-path /opt/blueprint/ckpts/sonic" in script
    assert "--embodiment-tag UNITREE_G1_SONIC" in script
    assert "--max-steps 500" in script
    assert "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL" in script
    assert "open_loop_qualification_passed" in script
    assert "g1_microwave_groot_open_loop_measurement.v1" in script
    assert "trained_checkpoint_open_loop_qualified" in script
    assert "maximum_error_ratio" in script
    assert "isaac_registered_transition_not_proven" in script
    assert component.REMOTE_GROOT_OVERLAY_ROOT in script
    assert (
        f"{component.REMOTE_GROOT_OVERLAY_ROOT}/gr00t/experiment/launch_finetune.py"
        in script
    )
    assert 'nested_checkpoint_processor_root = (' in script
    assert 'checkpoint_processor_root / \\"processor\\"' in script
    assert '"cosmos-reason2" in str(config.model_name).lower()' in script
    assert 'config.model.model_name = "nvidia/Cosmos-Reason2-2B"' in script
    assert "accelerate_other.is_deepspeed_available = lambda: False" in script
    assert "/opt/blueprint/models/cosmos-reason2-2b" in script
    assert "g1_microwave_finetune_local_cosmos_asset_invalid" in script
    assert 'PYTHONPATH="$GROOT_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"' in script
    assert "sealed_checkpoint_files_modified\": False" in script


def test_generated_component_accepts_exact_dataset_archive_members(tmp_path: Path) -> None:
    built = component.build_finetune_component(_dataset(tmp_path / "dataset"))
    script = built["script"]
    marker = 'python3 - "$DATASET" "$ARCHIVE_SHA" <<\'PY\'\n'
    extraction_source = script.split(marker, 1)[1].split("\nPY\n", 1)[0]
    extraction_root = tmp_path / "extracted"
    extraction_root.mkdir()

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            extraction_source,
            str(extraction_root),
            built["dataset_archive"]["sha256"],
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert {
        path.relative_to(extraction_root).as_posix()
        for path in extraction_root.rglob("*")
        if path.is_file()
    } == component.REQUIRED_DATASET_MEMBERS


def test_component_requires_live_aligned_isaac_training_episode(tmp_path: Path) -> None:
    built = component.build_finetune_component(_dataset(tmp_path / "dataset"))
    contract = built["live_aligned_training"]
    assert contract["required"] is True
    assert contract["same_session_live_start_required"] is True
    assert contract["exact_isaac_rigid_head_render_required"] is True
    assert len(contract["module_sha256"]) == 64
    assert contract["grasp_arc_module"] == component.GRASP_ARC_MODULE
    assert len(contract["grasp_arc_module_sha256"]) == 64
    script = built["script"]
    assert "prepare-actions" in script
    assert "render-isaac" in script
    assert "patch-dataset" in script
    assert "/workspace/initial_g1_sonic_state.json" in script
    assert component.REMOTE_GRASP_ARC_MODULE in script
    assert "--stage /workspace/kitchen/KitchenRoom.usd" in script
    assert (
        f"onnxruntime=={component.PINNED_ONNXRUNTIME_VERSION}"
        in script
    )
    assert (
        f"onnxruntime.__version__ == "
        f"{component.PINNED_ONNXRUNTIME_VERSION!r}"
        in script
    )


def test_component_overlay_patches_only_writable_copy(tmp_path: Path) -> None:
    script = component.build_finetune_component(_dataset(tmp_path / "dataset"))["script"]
    marker = (
        'python3 - "$GROOT_OVERLAY" '
        "/workspace/closed_loop_out/microwave_groot_overlay.json <<'PY'\n"
    )
    overlay_source = script.split(marker, 1)[1].split("\nPY\nset +e", 1)[0]
    compile(overlay_source, "microwave_groot_overlay.py", "exec")

    sealed_root = tmp_path / "sealed"
    model_source = sealed_root / "gr00t/model/gr00t_n1d7/gr00t_n1d7.py"
    setup_source = sealed_root / "gr00t/model/gr00t_n1d7/setup.py"
    launch_source = sealed_root / "gr00t/experiment/launch_finetune.py"
    model_source.parent.mkdir(parents=True)
    launch_source.parent.mkdir(parents=True)
    model_source.write_text(
        "def select(config):\n"
        '    if "nvidia/Cosmos-Reason2" in config.model_name '
        'or "Qwen/Qwen3-VL" in config.model_name:\n'
        "        return 1\n",
        encoding="utf-8",
    )
    setup_source.write_text(
        "from pathlib import Path\n"
        "class Setup:\n"
        "    def create(self):\n"
        "        if self.config.training.start_from_checkpoint is not None:\n"
        "            processor = AutoProcessor.from_pretrained(\n"
        "                self.config.training.start_from_checkpoint,\n"
        "            )\n"
        "        else:\n"
        "            processor = None\n"
        "        return processor\n",
        encoding="utf-8",
    )
    launch_source.write_text(
        "def configure(config):\n"
        '    config.model.model_name = "nvidia/Cosmos-Reason2-2B"\n'
        "    run(config)\n",
        encoding="utf-8",
    )
    model_before = model_source.read_bytes()
    setup_before = setup_source.read_bytes()
    model_source.chmod(0o444)
    setup_source.chmod(0o444)

    cache = tmp_path / "sealed-cache"
    snapshot = cache / "hub/model/snapshot"
    snapshot.mkdir(parents=True)
    local_model_root = tmp_path / "local-cosmos"
    for name in (
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        target = snapshot / name
        target.write_text(name, encoding="utf-8")
    local_model_root.symlink_to(snapshot, target_is_directory=True)

    patched_source = overlay_source.replace(
        'pathlib.Path("/opt/gr00t").resolve()',
        f"pathlib.Path({str(sealed_root)!r}).resolve()",
        1,
    ).replace(
        'pathlib.Path("/opt/blueprint/models/cosmos-reason2-2b")',
        f"pathlib.Path({str(local_model_root)!r})",
        1,
    ).replace(
        'pathlib.Path("/opt/blueprint/hf_home").resolve()',
        f"pathlib.Path({str(cache)!r}).resolve()",
        1,
    )
    overlay_root = tmp_path / "g1_microwave_groot_runtime"
    report_path = tmp_path / "overlay_report.json"
    completed = subprocess.run(
        [sys.executable, "-c", patched_source, str(overlay_root), str(report_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert model_source.read_bytes() == model_before
    assert setup_source.read_bytes() == setup_before
    copied_model = overlay_root / model_source.relative_to(sealed_root)
    copied_setup = overlay_root / setup_source.relative_to(sealed_root)
    copied_launch = overlay_root / launch_source.relative_to(sealed_root)
    assert '"cosmos-reason2" in str(config.model_name).lower()' in copied_model.read_text()
    assert 'checkpoint_processor_root / "processor"' in copied_setup.read_text()
    compile(copied_model.read_text(), str(copied_model), "exec")
    compile(copied_setup.read_text(), str(copied_setup), "exec")
    assert str(local_model_root) in copied_launch.read_text()
    assert "nvidia/Cosmos-Reason2-2B" not in copied_launch.read_text()
    assert (
        "accelerate_other.is_deepspeed_available = lambda: False"
        in copied_launch.read_text()
    )
    compile(copied_launch.read_text(), str(copied_launch), "exec")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["sealed_source_files_modified"] is False
    assert report["sealed_checkpoint_files_modified"] is False
    assert report["warm_start_weights_preserved"] is True
    assert report["offline_only"] is True
    assert len(report["local_model_files"]) == 4


def test_component_rejects_unreviewed_dataset_member(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset")
    (dataset / "secret.txt").write_text("no", encoding="utf-8")

    with pytest.raises(ValueError, match="dataset_member_unexpected"):
        component.build_finetune_component(dataset)
