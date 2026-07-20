from __future__ import annotations

from pathlib import Path

from blueprint_pipeline import g1_microwave_finetune_preflight as preflight


def test_finetune_preflight_pins_exact_groot_contract() -> None:
    assert preflight.PINNED_GROOT_N17_REVISION == (
        "e5749287857afd97b78f1147166137de29746392"
    )
    assert preflight.BASE_MODEL_ID == "nvidia/GR00T-N1.7-3B"
    assert preflight.SEALED_SONIC_WARM_START_PATH == "/opt/blueprint/ckpts/sonic"
    assert preflight.SEALED_SONIC_WARM_START_REPO == "LucaFrat/groot-bs16"
    assert preflight.SEALED_SONIC_WARM_START_REVISION == (
        "86b17337379926a8d8f1ad5c4580c7c33deeb49f"
    )
    assert preflight.EMBODIMENT_TAG == "UNITREE_G1_SONIC"
    assert preflight.EMBODIMENT_VALUE == "unitree_g1_sonic"
    assert preflight.EXPECTED_FRAME_COUNT == 176
    assert preflight.EXPECTED_ACTION_HORIZON == 40
    assert preflight.EXPECTED_EFFECTIVE_TIMESTEPS == 137


def test_bounded_finetune_plan_is_single_gpu_and_fail_closed(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    output = tmp_path / "checkpoint"
    argv = preflight.bounded_finetune_argv(
        dataset_path=dataset,
        output_dir=output,
    )

    assert argv[:2] == [
        "/opt/gr00t-venv/bin/python",
        "/opt/gr00t/gr00t/experiment/launch_finetune.py",
    ]
    pairs = dict(zip(argv[2:-1:2], argv[3:-1:2]))
    assert pairs["--base-model-path"] == "/opt/blueprint/ckpts/sonic"
    assert pairs["--dataset-path"] == str(dataset.resolve())
    assert pairs["--embodiment-tag"] == "UNITREE_G1_SONIC"
    assert pairs["--num-gpus"] == "1"
    assert pairs["--max-steps"] == "500"
    assert pairs["--global-batch-size"] == "1"
    assert pairs["--dataloader-num-workers"] == "0"
    assert pairs["--save-steps"] == "500"
    assert pairs["--save-total-limit"] == "1"
    assert pairs["--episode-sampling-rate"] == "1.0"
    assert pairs["--shard-size"] == "176"
    assert pairs["--num-shards-per-epoch"] == "1"
    assert pairs["--state-dropout-prob"] == "0.0"
    assert argv[-1] == "--save-only-model"
