"""Loopback community TRLC Pi0.5 host using an already staged immutable checkpoint.

No downloading, dependency installation, GPU allocation or default registration.
The later local-policy provisioning seam owns the process lifetime and budget.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time
from typing import Any
import uuid

import numpy as np

from .trlc_pi05_assets import verify_checkpoint_directory, verify_tokenizer_directory
from .trlc_pi05_policy import (
    MAX_MESSAGE_BYTES, WIRE_CAMERAS, IMAGE_SHAPES, TRLCPi05Binding, _array, _json, decode_array, encode_array,
)
from .policy_request_evidence import decode_scientific_wire


def decode_observation(payload: dict[str, Any]) -> dict[str, Any]:
    if (not isinstance(payload, dict) or set(payload) != {*WIRE_CAMERAS, "instruction", "state"}
            or not isinstance(payload.get("instruction"), str) or not payload["instruction"].strip()
            or len(payload["instruction"]) > 16384):
        raise ValueError("trlc_pi05_wire_observation_invalid")
    state = decode_array(payload["state"], (14,), "<f4")
    if not np.isfinite(state).all() or any(not 0 <= state[index] <= 1 for index in (6, 13)):
        raise ValueError("trlc_pi05_wire_state_invalid")
    return {"state": state, "instruction": payload["instruction"],
            **{name: decode_array(payload[name], shape, "|u1")
               for name, shape in zip(WIRE_CAMERAS, IMAGE_SHAPES, strict=True)}}


class TRLCPi05Backend:
    """Exactly saved pre/postprocessors; no relative-action factory defaults."""

    def __init__(self, *, model: Any, preprocess: Any, postprocess: Any):
        self.model, self.preprocess, self.postprocess = model, preprocess, postprocess

    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        import torch
        raw = {"observation.state": torch.from_numpy(observation["state"].copy()),
               "task": observation["instruction"]}
        for name in WIRE_CAMERAS:
            # The publisher performs pad/resize and [0,1]→[-1,1] exactly once.
            raw[name] = torch.from_numpy(observation[name].copy()).permute(2, 0, 1).float() / 255.0
        with torch.inference_mode():
            batch = self.preprocess(raw)
            normalized = self.model.predict_action_chunk(batch)
            # All28 dimensions are unnormalized before the client selects14.
            actions = self.postprocess(normalized)
        actions = actions.detach().to(device="cpu", dtype=torch.float32).numpy()
        if actions.shape == (1, 50, 28):
            actions = actions[0]
        return _array(actions, (50, 28), "trlc_pi05_backend_action_invalid")

    def reset(self) -> None:
        self.model.reset()
        for processor in (self.preprocess, self.postprocess):
            if callable(getattr(processor, "reset", None)):
                processor.reset()


def load_exact_state_dict(model: Any, state: dict) -> dict:
    """Only the tied alias declared by this exact safetensors header is added.

    Never call PI05Policy.from_pretrained: its broad exception handlers can
    return a randomly initialized policy after failed loading.
    """
    head = "model.paligemma_with_expert.paligemma.lm_head.weight"
    embedding = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    expected = model.state_dict()
    if set(state) != set(expected) - {embedding} or head not in state:
        raise ValueError("trlc_pi05_checkpoint_key_identity_changed")
    for name, tensor in state.items():
        if tuple(tensor.shape) != tuple(expected[name].shape):
            raise ValueError("trlc_pi05_checkpoint_tensor_shape_changed")
    mapped = {**state, embedding: state[head].clone()}
    result = model.load_state_dict(mapped, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("trlc_pi05_checkpoint_strict_load_failed")
    return {"strict_checkpoint_loaded": True, "saved_tensor_count": len(state),
            "runtime_tensor_count": len(expected), "alias": {embedding: head},
            "policy_inference_executed": False}


class TRLCPi05Server:
    def __init__(self, *, backend: Any, binding: TRLCPi05Binding, checkpoint_receipt: dict[str, Any]):
        binding.validate()
        self.backend = backend
        self.identity = json.loads(_json(binding.identity()))
        self.checkpoint_receipt = dict(checkpoint_receipt)
        self._lock = threading.Lock()
        self._session = None
        self._sequence = 0

    def handle(self, path: str, payload: dict[str, Any], *, session: str | None = None,
               sequence: str | None = None) -> dict[str, Any]:
        # Concurrent HTTP requests must never mix one episode's state with another.
        if not self._lock.acquire(blocking=False):
            raise ValueError("trlc_pi05_session_busy")
        try:
            if path == "/reset" and payload == {}:
                self._session = None
                self.backend.reset()
                self._session, self._sequence = uuid.uuid4().hex, 0
                return {"identity": self.identity, "session": self._session,
                        "checkpoint_bytes_verified": self.checkpoint_receipt["checkpoint_bytes_verified"],
                        "strict_checkpoint_loaded": self.checkpoint_receipt["strict_checkpoint_loaded"]}
            if (path != "/act" or self._session is None or session != self._session
                    or sequence != str(self._sequence + 1)):
                raise ValueError("trlc_pi05_session_or_sequence_invalid")
            observation = decode_observation(payload)
            started = time.monotonic()
            try:
                actions = _array(self.backend.predict(observation), (50, 28), "trlc_pi05_backend_action_invalid")
            except BaseException:
                self._session = None
                raise
            self._sequence += 1
            return {"actions": encode_array(actions), "dt_ms": (time.monotonic() - started) * 1000,
                    "session": self._session, "sequence": self._sequence}
        finally:
            self._lock.release()


def build_server(*, checkpoint: str | Path, tokenizer: str | Path,
                 binding: TRLCPi05Binding) -> TRLCPi05Server:
    binding.validate()
    deadline = os.environ.get("BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH")
    deadline = float(deadline) if deadline is not None else None
    receipt = verify_checkpoint_directory(Path(checkpoint), deadline_epoch=deadline)
    verify_tokenizer_directory(Path(tokenizer), deadline_epoch=deadline)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    import draccus
    import torch
    from safetensors.torch import load_file
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors

    root = Path(checkpoint).resolve(strict=True)
    with draccus.config_type("json"):
        config = draccus.parse(PreTrainedConfig, root / "config.json", args=[])
    config.device = "cuda"
    with torch.device("cuda"):
        model = PI05Policy(config)
    state = load_file(str(root / "model.safetensors"), device="cpu")
    receipt.update(load_exact_state_dict(model, state))
    del state
    model.eval()
    # A local pretrained_path loads the saved pipeline, not a synthesized one.
    preprocess, postprocess = make_pre_post_processors(config, str(root),
        preprocessor_overrides={"device_processor": {"device": "cuda"},
                               "tokenizer_processor": {"tokenizer_name": str(Path(tokenizer).resolve(strict=True))}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}})
    return TRLCPi05Server(backend=TRLCPi05Backend(model=model, preprocess=preprocess, postprocess=postprocess),
                          binding=binding, checkpoint_receipt=receipt)


def handler_for(runtime: TRLCPi05Server) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            pass  # Task prompts, arrays and local file paths do not enter HTTP logs.

        def do_POST(self) -> None:
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if self.path not in {"/act", "/reset"} or not 0 < size <= MAX_MESSAGE_BYTES:
                    raise ValueError("trlc_pi05_request_invalid")
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError("trlc_pi05_request_incomplete")
                payload = decode_scientific_wire(raw)
                response = runtime.handle(self.path, payload, session=self.headers.get("X-Blueprint-Session"),
                                          sequence=self.headers.get("X-Blueprint-Sequence"))
                status, body = 200, _json(response)
            except Exception:
                status, body = 400, _json({"error": "trlc_pi05_request_failed"})
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--port", type=int, default=8205)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    fields = json.loads(Path(args.binding).read_text())
    fields["camera_names"] = tuple(fields["camera_names"])
    binding = TRLCPi05Binding(**fields)
    binding.validate()
    if args.verify_only:
        print(_json(verify_checkpoint_directory(args.checkpoint)).decode())
        return
    runtime = build_server(checkpoint=args.checkpoint, tokenizer=args.tokenizer, binding=binding)
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(runtime)) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
