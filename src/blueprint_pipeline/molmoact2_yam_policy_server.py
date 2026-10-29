"""Loopback MolmoAct2 YAM host using an already staged immutable checkpoint.

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

from .molmoact2_yam_assets import NORM_TAG, verify_checkpoint_directory
from .molmoact2_yam_policy import (
    MAX_MESSAGE_BYTES, WIRE_CAMERAS, MolmoAct2YAMBinding, _array, _json, decode_array, encode_array,
)
from .policy_request_evidence import decode_scientific_wire


def decode_observation(payload: dict[str, Any]) -> dict[str, Any]:
    if (not isinstance(payload, dict)
            or set(payload) != {*WIRE_CAMERAS, "instruction", "state", "num_steps", "enable_cuda_graph"}
            or not isinstance(payload.get("instruction"), str) or not payload["instruction"].strip()
            or len(payload["instruction"]) > 16384
            or type(payload.get("num_steps")) is not int or payload["num_steps"] != 10
            or payload.get("enable_cuda_graph") is not False):
        raise ValueError("molmoact2_wire_observation_invalid")
    state = decode_array(payload["state"], (14,), "<f4")
    if not np.isfinite(state).all() or any(not 0 <= state[index] <= 1 for index in (6, 13)):
        raise ValueError("molmoact2_wire_state_invalid")
    return {"state": state, "instruction": payload["instruction"],
            **{name: decode_array(payload[name], (360, 640, 3), "|u1") for name in WIRE_CAMERAS}}


class MolmoAct2YAMBackend:
    """Publisher predict_action API; normalization stays entirely in that API."""

    def __init__(self, *, model: Any, processor: Any):
        self.model, self.processor = model, processor

    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        from PIL import Image
        import torch

        with torch.inference_mode():
            output = self.model.predict_action(
                processor=self.processor,
                images=[Image.fromarray(observation[name]) for name in WIRE_CAMERAS],
                task=observation["instruction"], state=observation["state"], norm_tag=NORM_TAG,
                inference_action_mode="continuous", enable_depth_reasoning=False,
                num_steps=10, n_action_steps=30, normalize_language=True, enable_cuda_graph=False,
            )
        actions = output.actions
        if torch.is_tensor(actions):
            actions = actions.detach().to(device="cpu", dtype=torch.float32).numpy()
        actions = np.asarray(actions)
        if actions.shape == (1, 30, 14):
            actions = actions[0]
        return _array(actions, (30, 14), "molmoact2_backend_action_invalid")

    def reset(self) -> None:
        # n_obs_steps=1; each predict_action receives all three current images,
        # state and task. No action queue, depth cache or CUDA graph is retained.
        pass


class MolmoAct2YAMServer:
    def __init__(self, *, backend: Any, binding: MolmoAct2YAMBinding, checkpoint_receipt: dict[str, Any]):
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
            raise ValueError("molmoact2_session_busy")
        try:
            if path == "/reset" and payload == {}:
                self._session = None
                self.backend.reset()
                self._session, self._sequence = uuid.uuid4().hex, 0
                return {"identity": self.identity, "session": self._session,
                        "checkpoint_bytes_verified": self.checkpoint_receipt["checkpoint_bytes_verified"]}
            if (path != "/act" or self._session is None or session != self._session
                    or sequence != str(self._sequence + 1)):
                raise ValueError("molmoact2_session_or_sequence_invalid")
            observation = decode_observation(payload)
            started = time.monotonic()
            try:
                actions = _array(self.backend.predict(observation), (30, 14), "molmoact2_backend_action_invalid")
            except BaseException:
                self._session = None
                raise
            self._sequence += 1
            return {"actions": encode_array(actions), "dt_ms": (time.monotonic() - started) * 1000,
                    "session": self._session, "sequence": self._sequence}
        finally:
            self._lock.release()


def build_server(*, checkpoint: str | Path, binding: MolmoAct2YAMBinding) -> MolmoAct2YAMServer:
    binding.validate()
    receipt = verify_checkpoint_directory(checkpoint)
    # This entrypoint runs in its own interpreter. No credentials/network lookup
    # or upstream server's mutable-cache BF16 patch is needed for FP32 inference.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import torch

    root = str(Path(checkpoint).resolve(strict=True))
    processor = AutoProcessor.from_pretrained(root, trust_remote_code=True, local_files_only=True,
                                              extra_special_tokens={})
    model = AutoModelForImageTextToText.from_pretrained(root, trust_remote_code=True, local_files_only=True,
                                                       torch_dtype=torch.float32).to("cuda").eval()
    return MolmoAct2YAMServer(backend=MolmoAct2YAMBackend(model=model, processor=processor),
                             binding=binding, checkpoint_receipt=receipt)


def handler_for(runtime: MolmoAct2YAMServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            pass  # Task prompts, arrays and local file paths do not enter HTTP logs.

        def do_POST(self) -> None:
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if self.path not in {"/act", "/reset"} or not 0 < size <= MAX_MESSAGE_BYTES:
                    raise ValueError("molmoact2_request_invalid")
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError("molmoact2_request_incomplete")
                payload = decode_scientific_wire(raw)
                response = runtime.handle(self.path, payload, session=self.headers.get("X-Blueprint-Session"),
                                          sequence=self.headers.get("X-Blueprint-Sequence"))
                status, body = 200, _json(response)
            except Exception:
                status, body = 400, _json({"error": "molmoact2_request_failed"})
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
    parser.add_argument("--port", type=int, default=8202)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    fields = json.loads(Path(args.binding).read_text())
    fields["camera_names"] = tuple(fields["camera_names"])
    binding = MolmoAct2YAMBinding(**fields)
    binding.validate()
    if args.verify_only:
        print(_json(verify_checkpoint_directory(args.checkpoint)).decode())
        return
    runtime = build_server(checkpoint=args.checkpoint, binding=binding)
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(runtime)) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
