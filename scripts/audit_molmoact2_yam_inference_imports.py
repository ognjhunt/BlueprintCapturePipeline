"""Local CPU check of verified MolmoAct2 code, tokenizer, config and API surface.

No model is constructed, weights loaded, inference requested, or GPU used.
Socket connections are blocked before importing Transformers/checkpoint code.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import socket
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-metadata", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from blueprint_pipeline.molmoact2_yam_dependencies import verified_dependency_manifest

    manifest = verified_dependency_manifest()
    checkpoint = args.checkpoint_metadata.resolve(strict=True)
    expected_files = {row["filename"] for row in manifest["checkpoint_audit_inputs"]}
    entries = list(checkpoint.rglob("*"))
    if (any(path.is_symlink() for path in entries)
            or {path.relative_to(checkpoint).as_posix() for path in entries if path.is_file()} != expected_files):
        raise ValueError("molmoact2_cpu_audit_input_inventory_mismatch")
    for row in manifest["checkpoint_audit_inputs"]:
        path = checkpoint / row["filename"]
        if path.is_symlink() or path.stat().st_size != row["size_bytes"]:
            raise ValueError("molmoact2_cpu_audit_input_invalid")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("molmoact2_cpu_audit_input_digest_mismatch")
    cache = args.cache.resolve()
    if cache == checkpoint or checkpoint in cache.parents:
        raise ValueError("molmoact2_cpu_audit_cache_must_be_outside_checkpoint")
    os.environ.update(HF_HOME=str(cache), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    attempts = []

    def reject_network(*_args, **_kwargs):
        attempts.append("socket_connect_attempt")
        raise RuntimeError("molmoact2_cpu_audit_network_forbidden")

    socket.socket.connect = reject_network
    socket.socket.connect_ex = reject_network
    socket.create_connection = reject_network
    import numpy as np
    from PIL import Image
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    if torch.__version__ != "2.11.0+cpu" or torch.cuda.is_available() or transformers.__version__ != "4.57.6":
        raise RuntimeError("molmoact2_cpu_audit_requires_pinned_cpu_environment")
    config = AutoConfig.from_pretrained(str(checkpoint), trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True,
                                              local_files_only=True, extra_special_tokens={})
    model_class = get_class_from_dynamic_module(config.auto_map["AutoModelForImageTextToText"],
                                                str(checkpoint), local_files_only=True)
    module = importlib.import_module(model_class.__module__)
    stats = module._RobotStats(json.loads((checkpoint / "norm_stats.json").read_text()))
    tag = stats.validate_tag("yam_dual_molmoact2")
    state = np.zeros(14, dtype=np.float32)
    state[[6, 13]] = [0.25, 0.75]
    normalized = np.clip(stats.normalize_state(state, tag), -1, 1)
    diagnostic_actions = np.zeros((30, 14), dtype=np.float32)
    diagnostic_actions[:, [6, 13]] = [0.25, 0.75]
    physical = stats.unnormalize_action(diagnostic_actions, tag)
    metadata = stats.get_metadata(tag)
    text = module._build_robot_text(
        task=module._normalize_question_text("Move the object to the target."), style="robot_action",
        discrete_state_string=module._build_discrete_state_string(normalized, int(config.num_state_tokens)),
        setup_type=str(metadata["setup_type"]), control_mode=str(metadata["control_mode"]),
        add_setup_tokens=bool(config.add_setup_tokens), add_control_tokens=bool(config.add_control_tokens),
        num_images=3,
    )
    images = [Image.new("RGB", (640, 360), color) for color in [(20, 40, 60), (40, 60, 20), (60, 20, 40)]]
    inputs = processor(text=text, images=images, return_tensors="pt")
    # Validate the exact wrapper call signature without invoking the model.
    signature = inspect.signature(model_class.predict_action)
    signature.bind(None, processor=processor, images=images, task="Move the object to the target.",
                   state=state, norm_tag=tag, inference_action_mode="continuous", enable_depth_reasoning=False,
                   num_steps=10, n_action_steps=30, normalize_language=True, enable_cuda_graph=False)
    assert stats.get_action_dim(tag) == stats.get_state_dim(tag) == 14
    assert config.max_action_horizon == 30 and config.n_obs_steps == 1
    assert np.isfinite(normalized).all() and np.isfinite(physical).all()
    assert np.array_equal(normalized[[6, 13]], state[[6, 13]])
    assert np.array_equal(physical[:, [6, 13]], diagnostic_actions[:, [6, 13]])
    assert all(torch.isfinite(value).all() for value in inputs.values() if torch.is_tensor(value))
    assert not attempts
    source_files = {}
    expected = {row["filename"]: row["sha256"] for row in manifest["checkpoint_import_sources"]}
    for name, loaded in list(sys.modules.items()):
        filename = getattr(loaded, "__file__", None)
        if not name.startswith("transformers_modules.") or not filename:
            continue
        path = Path(filename)
        if path.name in expected:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == expected[path.name]
            source_files[path.name] = digest
    assert source_files == expected
    receipt = {
        "schema": "molmoact2_yam_cpu_dependency_audit.v1", "status": "passed",
        "python_version": sys.version.split()[0], "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "checkpoint_transformers_metadata": "5.3.0",
        "processor_class": type(processor).__name__, "model_class_imported": model_class.__name__,
        "predict_action_signature": str(signature), "input_shapes": {
            key: list(value.shape) for key, value in inputs.items() if torch.is_tensor(value)},
        "normalized_state_shape": list(normalized.shape), "physical_action_shape": list(physical.shape),
        "independent_gripper_channels_preserved": True, "network_attempts": len(attempts),
        "source_files": source_files, "model_revision": manifest["model_revision"],
        "installed_packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "cuda_inference_performed": False, "model_constructed": False, "model_weights_loaded": False,
        "policy_queried": False, "task_executed": False, "vram_requirement_measured": False,
        "claim_ceiling": "cpu_import_config_tokenizer_preprocessing_api_only",
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "source_files": len(source_files), "input_shapes": receipt["input_shapes"]}))


if __name__ == "__main__":
    main()
