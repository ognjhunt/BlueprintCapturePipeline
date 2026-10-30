"""Offline TRLC source/meta-shape/saved-processor proof; no weight inference.

Requires the reviewed G1 CPU dependency environment and retained small checkpoint
files plus the bounded safetensors header. Tokenization is deliberately stubbed:
this audit does not fetch gated assets or claim their current host availability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-inputs", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from blueprint_pipeline.trlc_pi05_assets import release_inventory, MODEL_REVISION
    from blueprint_pipeline.trlc_pi05_source import stage_source, extract_source, SOURCE_ARCHIVE, SOURCE_MANIFEST
    args.work.mkdir(parents=True, exist_ok=True)
    stage_source(args.work)
    publisher = extract_source(args.work / SOURCE_ARCHIVE, args.work / SOURCE_MANIFEST, args.work / "source")
    sys.path.insert(0, str(publisher / "lerobot/src"))
    for row in release_inventory()["files"]:
        if row["path"] == "model.safetensors":
            continue
        raw = (args.audit_inputs / row["path"]).read_bytes()
        assert len(raw) == row["size_bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"]
    header_raw = args.header.read_bytes()
    assert hashlib.sha256(header_raw).hexdigest() == "91e40d60ed3546b90593d90d06677cb5152587d0d9dc7501fb63f0e3e9ab3186"
    header = json.loads(header_raw)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      HF_HOME=str(args.work / "empty-cache"))
    attempts = []
    def deny(*_, **__):
        attempts.append(True)
        raise RuntimeError("trlc_cpu_proof_network_forbidden")
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    import draccus
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    from blueprint_pipeline.trlc_pi05_policy_server import TRLCPi05Backend, load_exact_state_dict
    from blueprint_pipeline.trlc_pi05_policy import WIRE_CAMERAS, IMAGE_SHAPES, JOINTS, GRIPPERS, decode_actions
    assert torch.__version__ == "2.10.0+cpu" and not torch.cuda.is_available()
    with draccus.config_type("json"):
        config = draccus.parse(PreTrainedConfig, args.audit_inputs / "config.json", args=[])
    config.device = "meta"
    with torch.device("meta"):
        meta = PI05Policy(config)
    state = {name: torch.empty(row["shape"], device="meta", dtype={"BF16": torch.bfloat16, "F32": torch.float32}[row["dtype"]])
             for name, row in header.items() if name != "__metadata__"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        key_proof = load_exact_state_dict(meta, state)
    assert key_proof["saved_tensor_count"] == 812 and key_proof["runtime_tensor_count"] == 813
    assert all(header["__metadata__"][target] == origin for target, origin in key_proof["alias"].items())
    del meta, state
    config.device = "cpu"
    prompts = []
    class StubTokenizer:
        padding_side = "right"
        def __call__(self, text, **kwargs):
            prompts.extend(text)
            return {"input_ids": torch.zeros((len(text), kwargs["max_length"]), dtype=torch.long),
                    "attention_mask": torch.ones((len(text), kwargs["max_length"]), dtype=torch.long)}
    pre, post = make_pre_post_processors(config, str(args.audit_inputs),
        preprocessor_overrides={"device_processor": {"device": "cpu"},
                               "tokenizer_processor": {"tokenizer": StubTokenizer()}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}})
    normalized = torch.linspace(-1, 1, 50 * 28).reshape(1, 50, 28)
    received, images = [], []
    class FixtureModel:
        def predict_action_chunk(self, batch):
            received.append(batch)
            image_self = type("ImageOnly", (), {"config": config, "parameters": lambda _: iter([torch.zeros(1)])})()
            images.extend(PI05Policy._preprocess_images(image_self, batch)[0])
            return normalized
        def reset(self):
            pass
    backend = TRLCPi05Backend(model=FixtureModel(), preprocess=pre, postprocess=post)
    state_stats = load_file(str(args.audit_inputs / "policy_preprocessor_step_2_normalizer_processor.safetensors"))
    raw_state = ((state_stats["observation.state.q01"] + state_stats["observation.state.q99"]) / 2).numpy()
    raw_state[[6, 13]] = [.2, .8]
    observation = {"state": raw_state, "instruction": "Move the object to the target.",
        **{name: np.full(shape, value, dtype=np.uint8) for name, shape, value in zip(WIRE_CAMERAS, IMAGE_SHAPES, (51, 127, 204), strict=True)}}
    actual = backend.predict(observation)
    stats = load_file(str(args.audit_inputs / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    expected = ((normalized + 1) / 2 * (stats["action.q99"] - stats["action.q01"]) + stats["action.q01"])[0].numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert actual.shape == (50, 28)
    physical, _ = decode_actions(actual, joint_names=list(JOINTS), gripper_names=list(GRIPPERS))
    np.testing.assert_allclose(physical[:, :12], actual[:, [0,1,2,3,4,5,7,8,9,10,11,12]])
    centers = [float(image[0, 0, 112, 112]) for image in images]
    np.testing.assert_allclose(centers, np.array([51, 127, 204]) / 255 * 2 - 1, atol=1e-6)
    assert all(tuple(image.shape) == (1, 3, 224, 224) for image in images)
    assert prompts[0].startswith("Task: Move the object to the target., State:")
    assert not attempts
    receipt = {"schema": "trlc_pi05_cpu_adapter_audit.v1", "status": "passed", "checkpoint_revision": MODEL_REVISION,
        "source_directory_verified": True, "saved_preprocessor_steps": [type(step).__name__ for step in pre.steps],
        "saved_postprocessor_steps": [type(step).__name__ for step in post.steps],
        "meta_saved_tensor_count": 812, "meta_runtime_tensor_count": 813, "tied_alias": key_proof["alias"],
        "physical_action_shape": [50, 14], "full_unnormalized_action_shape": list(actual.shape),
        "image_center_values": centers, "camera_order": list(WIRE_CAMERAS), "prepared_prompt": prompts[0],
        "network_attempts": len(attempts), "actual_tokenizer_executed": False,
        "full_checkpoint_verified": False, "model_weights_loaded": False, "policy_inference_executed": False,
        "native_execution_validated": False, "peak_gpu_memory_measured": False,
        "claim_ceiling": "cpu_source_meta_shapes_saved_processors_with_stub_tokenizer_and_model"}
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "meta_tensors": 813, "full_action_shape": list(actual.shape)}))


if __name__ == "__main__":
    main()
