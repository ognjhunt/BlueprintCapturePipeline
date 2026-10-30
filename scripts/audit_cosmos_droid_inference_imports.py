"""Run the pinned Cosmos preprocessing surface on CPU, with network denied."""

import sys
import os
import json
import socket
import hashlib
from pathlib import Path
from types import SimpleNamespace


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Offline import/tokenizer/preprocessing proof; never load policy weights."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint-metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args_cli = parser.parse_args()
    root, checkpoint = args_cli.output_root.resolve(), args_cli.checkpoint_metadata.resolve()
    root.mkdir(parents=True, exist_ok=True)
    from blueprint_pipeline.cosmos_droid_source import verify_source_directory, SOURCE_MANIFEST
    from blueprint_pipeline.cosmos_nano_beta_runtime import (
        snapshot_manifest_path,
        verified_snapshot_manifest,
    )

    vendor = Path(snapshot_manifest_path()).parent
    source = verify_source_directory(args_cli.source, vendor / SOURCE_MANIFEST)
    manifest = verified_snapshot_manifest(snapshot_manifest_path())
    metadata_rows = [
        row for row in manifest["required_files"] if not row["path"].endswith(".safetensors")
    ]
    assert not list(checkpoint.rglob("*.safetensors"))
    for row in metadata_rows:
        payload = (checkpoint / row["path"]).read_bytes()
        assert (
            len(payload) == row["size_bytes"]
            and hashlib.sha256(payload).hexdigest() == row["sha256"]
        )
    sys.path.insert(0, str(source))
    os.environ.update(
        COSMOS_TRAINING="false",
        COSMOS_DEVICE="cpu",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        MPLCONFIGDIR=str(root / "mpl"),
    )
    sys.dont_write_bytecode = True
    connections = []

    def deny(*a, **k):
        connections.append(True)
        raise RuntimeError("network disallowed in offline proof")

    socket.socket.connect = deny
    socket.create_connection = deny
    import numpy as np
    import torch
    from cosmos_framework.scripts.action_policy_server_robolab import (
        RobolabServerArgs,
        RobolabPolicyConfig,
        RobolabPolicyService,
    )
    from cosmos_framework.inference.inference import (
        _bundled_processor_serves_node,
        _point_tokenizer_node_at_dir,
    )
    from cosmos_framework.data.generator.processors import build_processor
    from openpi_client import msgpack_numpy

    args = RobolabServerArgs(
        checkpoint_path=str(checkpoint),
        hf_revision="805c0d6d46196ecdc789213a6de72303c383895b",
        format_prompt_as_json=False,
        output_dir=root / "setup-output",
        deterministic_seed=True,
    )
    service = object.__new__(RobolabPolicyService)
    setup = service._build_setup_args(args)
    modeldict = setup.load_model_config_dict()
    tokenizer = modeldict["config"]["vlm_config"]["tokenizer"]
    assert _bundled_processor_serves_node(tokenizer, checkpoint)
    assert _point_tokenizer_node_at_dir(tokenizer, checkpoint)
    processor = build_processor(
        tokenizer_type=str(checkpoint), cache_dir=str(root / "processor-cache")
    )
    ids = processor.tokenizer("Pick up the cup.")["input_ids"]
    assert len(ids) > 0
    service.model = SimpleNamespace(
        config=SimpleNamespace(max_action_dim=modeldict["config"]["max_action_dim"])
    )
    service._transform, inferred = service._build_transform(None, args)
    service.cfg = RobolabPolicyConfig(
        **{key: getattr(args, key) for key in RobolabPolicyConfig.__dataclass_fields__}
    )
    obs = {
        "observation/wrist_image_left": np.full((224, 224, 3), 30, np.uint8),
        "observation/exterior_image_1_left": np.full((224, 224, 3), 90, np.uint8),
        "observation/exterior_image_2_left": np.full((224, 224, 3), 150, np.uint8),
        "observation/joint_position": np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], np.float32),
        "observation/gripper_position": np.array([0.25], np.float32),
        "prompt": "Pick up the cup.",
    }
    wire = msgpack_numpy.unpackb(msgpack_numpy.packb(obs))
    assert np.array_equal(wire["observation/joint_position"], obs["observation/joint_position"])
    sample = service._build_sample(wire)
    assert isinstance(sample["ai_caption"], str) and not sample["ai_caption"].startswith("{")
    assert tuple(sample["action"].shape) == (33, 64)
    assert torch.allclose(
        sample["action"][0, :7], torch.from_numpy(obs["observation/joint_position"])
    )
    assert sample["action"][0, 7].item() == 0.75
    assert torch.count_nonzero(sample["action"][1:]) == 0
    assert not connections
    result = {
        "schema": "cosmos_nano_cpu_import_preprocessing_proof.v1",
        "source_revision": "2f603cb114ff8b335e116060444d0b6caee3a85e",
        "model_revision": args.hf_revision,
        "python_version": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "server_imported": True,
        "tokenizer_type": type(processor).__name__,
        "token_count": len(ids),
        "sample_shapes": {k: list(v.shape) for k, v in sample.items() if hasattr(v, "shape")},
        "plain_prompt": True,
        "current_state_row_verified": True,
        "gripper_inversion_verified": True,
        "openpi_numpy2_wire_roundtrip_verified": True,
        "network_connections_attempted": len(connections),
        "model_constructed": False,
        "weights_loaded": False,
        "candidate_inference_performed": False,
        "native_cuda_kernels_tested": False,
        "note": "Official uv.lock overrides openpi-client NumPy<2 metadata with numpy2.2.6; exact wire roundtrip tested, pip check reports this known publisher override.",
    }
    (root / "cpu-proof.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
