"""Pinned local OpenWAM server entrypoint; no allocation, download or registration.

The separate policy interpreter owns Torch/OpenWAM; Isaac imports only the thin
NumPy/Pillow/WebSocket client. One connection owns one temporal policy session.
This entrypoint is intentionally not wired into a paid bundle or beta catalog.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any

from .openwam_piper_assets import verify_checkpoint_directory, verify_source_checkout
from .openwam_piper_policy import (
    MAX_MESSAGE_BYTES, WIRE_CAMERAS, OpenWAMPiperBinding, _json, _vector,
)
from .policy_request_evidence import decode_scientific_wire


def validate_wire_observation(value: dict[str, Any]) -> dict[str, Any]:
    """Keep the server boundary limited to declared sensor inputs and task text."""
    from PIL import Image

    if (set(value) != {"type", "images", "state", "prompt"} or value.get("type") != "obs"
            or not isinstance(value.get("images"), dict) or set(value["images"]) != set(WIRE_CAMERAS)
            or not isinstance(value.get("prompt"), str) or not value["prompt"].strip()
            or len(value["prompt"]) > 16512):
        raise ValueError("openwam_wire_observation_invalid")
    state = _vector(value["state"], 20, "openwam_wire_state_invalid")
    if any(not 0 <= state[index] <= 1 for index in (9, 19)):
        raise ValueError("openwam_wire_gripper_invalid")
    for encoded in value["images"].values():
        if not isinstance(encoded, str):
            raise ValueError("openwam_wire_image_invalid")
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG" or image.mode != "RGB" or any(not 1 <= side <= 4096 for side in image.size):
                raise ValueError("openwam_wire_image_invalid")
            image.verify()
    return {"images": value["images"], "state": state.tolist(), "prompt": value["prompt"]}


class OpenWAMPiperServer:
    """Add exact release identity to the publisher's existing predict/reset API."""

    def __init__(self, *, backend: Any, binding: OpenWAMPiperBinding, checkpoint_receipt: dict[str, Any]):
        binding.validate()
        self.backend, self.binding = backend, binding
        self.checkpoint_receipt = checkpoint_receipt
        self._session = threading.Lock()

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(message, dict):
            raise ValueError("openwam_message_invalid")
        if message == {"type": "ping"}:
            return {"type": "pong", "blueprint_identity": self.binding.identity(),
                    "checkpoint_bytes_verified": self.checkpoint_receipt["checkpoint_bytes_verified"]}
        if message == {"type": "reset"}:
            self.backend.reset()
            return {"type": "reset_ack"}
        observation = validate_wire_observation(message)
        # Publisher owns PNG decoding, 384x320 L-shaped view assembly, normalized
        # 80D scatter/gather and its 32-row action buffer. Return physical EEF20.
        response = self.backend.predict(observation)
        if not isinstance(response, dict):
            raise ValueError("openwam_backend_response_invalid")
        return {**response, "type": "action"}

    def serve_connection(self, socket: Any) -> None:
        if not self._session.acquire(blocking=False):
            socket.close(code=1013, reason="policy session already active")
            return
        try:
            self.backend.reset()
            for message in socket:
                try:
                    if not isinstance(message, str) or len(message.encode()) > MAX_MESSAGE_BYTES:
                        raise ValueError("openwam_message_invalid")
                    response = self.handle(decode_scientific_wire(message.encode()))
                    socket.send(_json(response))
                except Exception:
                    # Do not leak task text, paths or incidental exception payloads.
                    socket.send(_json({"type": "error", "code": "openwam_request_failed"}))
                    break
        finally:
            try:
                self.backend.reset()
            finally:
                self._session.release()
                socket.close()


def build_server(*, checkpoint: str | Path, source_root: str | Path,
                 binding: OpenWAMPiperBinding) -> OpenWAMPiperServer:
    """Load exact local bytes through the upstream supported checkpoint builder."""
    binding.validate()
    source = verify_source_checkout(source_root)
    receipt = verify_checkpoint_directory(checkpoint)
    # A self-contained checkpoint requires no hub/network/credential discovery.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(source))
    from openwam.deploy import server as upstream
    from omegaconf import OmegaConf

    if source not in Path(upstream.__file__).resolve().parents:
        raise ValueError("openwam_import_source_mismatch")
    settings = OmegaConf.create({
        "inference": {"denoise_steps": 10, "inference_mode": "sync", "denoise_mode": "sync",
                      "inference_horizon": binding.execution_horizon},
        "optimization": {"decode_video": False, "compile": {"enabled": False},
                         "dit_cache": {"enabled": False}, "prompt_embed_cache": {"enabled": True, "maxsize": 32}},
    })
    backend = upstream.build_server_from_config(settings, str(Path(checkpoint).resolve()), device="cuda",
                                                ckpt_name="checkpoint_step_30000.safetensors")
    return OpenWAMPiperServer(backend=backend, binding=binding, checkpoint_receipt=receipt)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--port", type=int, default=8848)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    data = json.loads(Path(args.binding).read_text())
    data["camera_names"] = tuple(data["camera_names"])
    binding = OpenWAMPiperBinding(**data)
    binding.validate()
    if args.verify_only:
        verify_source_checkout(args.source_root)
        print(_json(verify_checkpoint_directory(args.checkpoint)))
        return
    runtime = build_server(checkpoint=args.checkpoint, source_root=args.source_root, binding=binding)
    from websockets.sync.server import serve

    try:
        with serve(runtime.serve_connection, "127.0.0.1", args.port, max_size=MAX_MESSAGE_BYTES,
                   compression=None) as listener:
            listener.serve_forever()
    finally:
        runtime.backend.shutdown()


if __name__ == "__main__":
    main()
