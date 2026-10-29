"""Local CPU check of real PiPER inference imports and staged small assets.

Requires the pinned dependency environment with CPU Torch substituted. Does not
load checkpoint weights, construct the model, contact a service, or use a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    checkpoint = args.checkpoint_metadata.resolve(strict=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      DIFFSYNTH_SKIP_DOWNLOAD="true", WAM_ATTENTION_IMPL="sdpa",
                      DIFFSYNTH_ATTENTION_IMPLEMENTATION="torch")
    network_attempts = []

    def reject_network(*_args, **_kwargs):
        network_attempts.append("socket_connect_attempt")
        raise RuntimeError("cpu_import_audit_network_forbidden")

    socket.socket.connect = reject_network
    socket.socket.connect_ex = reject_network
    socket.create_connection = reject_network
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from PIL import Image

    if torch.__version__ != "2.7.1+cpu" or torch.cuda.is_available():
        raise RuntimeError("cpu_import_audit_requires_pinned_cpu_torch")
    modules = [
        "openwam.deploy.server", "openwam.deploy.model_loader", "openwam.deploy.policy",
        "openwam.deploy.executors.sync_executor", "openwam.model.video_backbone.wan.pipeline_builder",
        "openwam.model.video_backbone.wan.models.dit", "openwam.model.video_backbone.wan.models.vae",
        "openwam.model.video_backbone.wan.models.text_encoder",
    ]
    for name in modules:
        module = importlib.import_module(name)
        if source not in Path(module.__file__).resolve().parents:
            raise RuntimeError("cpu_import_audit_source_mismatch")
    from openwam.deploy.model_loader import _build_normalizer
    from openwam.deploy.obs_preprocess import ObsPreprocessor
    from openwam.model import resolve_architecture_config
    from openwam.model.video_backbone.wan.models.text_encoder import HuggingfaceTokenizer

    config = OmegaConf.load(checkpoint / "config.yaml")
    architecture = resolve_architecture_config(config.model)
    tokenizer = HuggingfaceTokenizer(str(checkpoint / "tokenizer/google/umt5-xxl"),
                                    seq_len=512, clean="whitespace", local_files_only=True)
    tokens = tokenizer(["Move the object to the target."])
    normalizer = _build_normalizer(config, str(checkpoint))
    normalized = normalizer.normalize(np.zeros((1, 20), dtype=np.float32))
    physical = normalizer.unnormalize(np.zeros((1, 80), dtype=np.float32))
    image = Image.new("RGB", (320, 176), (20, 40, 60))
    obs = ObsPreprocessor.from_cfg(config).preprocess({
        "images": {name: image for name in ("head_camera", "left_wrist_camera", "right_wrist_camera")},
        "state": [0.0] * 20, "prompt": "Move the object to the target.",
    })
    assert list(tokens.shape) == [1, 512]
    assert list(normalized.shape) == [1, 80] and list(physical.shape) == [1, 20]
    assert np.isfinite(normalized).all() and np.isfinite(physical).all()
    assert obs["image"].size == (320, 384)
    assert not network_attempts
    imported_files = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("openwam") and path and str(path).endswith(".py"):
            path = Path(path).resolve()
            imported_files[path.relative_to(source).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "schema": "openwam_piper_cpu_dependency_audit.v1", "status": "passed",
        "python_version": sys.version.split()[0], "torch_version": torch.__version__,
        "architecture": architecture.registry_name, "token_shape": list(tokens.shape),
        "normalized_state_shape": list(normalized.shape), "physical_action_shape": list(physical.shape),
        "observation_canvas_width_height": list(obs["image"].size),
        "network_attempts": len(network_attempts), "source_files": imported_files,
        "installed_packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "cuda_inference_performed": False, "model_weights_loaded": False,
        "policy_queried": False, "task_executed": False, "vram_requirement_measured": False,
        "claim_ceiling": "cpu_import_config_tokenizer_preprocessing_only",
    }
    args.output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "passed", "source_files": len(imported_files),
                      "claim_ceiling": receipt["claim_ceiling"]}))


if __name__ == "__main__":
    main()
