"""Pinned local OpenWAM server entrypoint; no allocation, download or registration.

The separate policy interpreter owns Torch/OpenWAM; Isaac imports only the thin
NumPy/Pillow/WebSocket client. One connection owns one temporal policy session.
The local runtime bundle uses this entrypoint after bounded artifact staging.
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


def validate_wire_observation(value: dict[str, Any], *, cameras=WIRE_CAMERAS, state_width=20,
                              gripper_indices=(9, 19), gripper_bounds=(0, 1), image_size=None) -> dict[str, Any]:
    """Keep the server boundary limited to declared sensor inputs and task text."""
    from PIL import Image

    if (set(value) != {"type", "images", "state", "prompt"} or value.get("type") != "obs"
            or not isinstance(value.get("images"), dict) or set(value["images"]) != set(cameras)
            or not isinstance(value.get("prompt"), str) or not value["prompt"].strip()
            or len(value["prompt"]) > 16512):
        raise ValueError("openwam_wire_observation_invalid")
    state = _vector(value["state"], state_width, "openwam_wire_state_invalid")
    if any(not gripper_bounds[0] <= state[index] <= gripper_bounds[1] for index in gripper_indices):
        raise ValueError("openwam_wire_gripper_invalid")
    for encoded in value["images"].values():
        if not isinstance(encoded, str):
            raise ValueError("openwam_wire_image_invalid")
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if (image.format != "PNG" or image.mode != "RGB" or any(not 1 <= side <= 4096 for side in image.size)
                    or (image_size is not None and image.size != image_size)):
                raise ValueError("openwam_wire_image_invalid")
            image.verify()
    return {"images": value["images"], "state": state.tolist(), "prompt": value["prompt"]}


class OpenWAMPiperServer:
    """Add exact release identity to the publisher's existing predict/reset API."""

    validate_observation = staticmethod(validate_wire_observation)

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
        observation = self.validate_observation(message)
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
                 binding: OpenWAMPiperBinding, source_manifest: str | Path | None = None) -> OpenWAMPiperServer:
    return _build_server(checkpoint=checkpoint, source_root=source_root, binding=binding, source_manifest=source_manifest,
        checkpoint_verifier=verify_checkpoint_directory, server_type=OpenWAMPiperServer,
        checkpoint_name="checkpoint_step_30000.safetensors")


def _build_server(*, checkpoint, source_root, binding, source_manifest, checkpoint_verifier, server_type, checkpoint_name):
    """Load exact local bytes through the upstream supported checkpoint builder."""
    binding.validate()
    if source_manifest is None:
        source = verify_source_checkout(source_root)
    else:
        from .openwam_piper_source import verify_source_directory
        source = verify_source_directory(source_root, source_manifest)
    deadline = os.environ.get("BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH")
    receipt = checkpoint_verifier(checkpoint, **({"deadline_epoch": float(deadline)} if deadline else {}))
    # A self-contained checkpoint requires no hub/network/credential discovery.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
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
                                                ckpt_name=checkpoint_name)
    return server_type(backend=backend, binding=binding, checkpoint_receipt=receipt)


def main(argv: list[str] | None = None) -> None:
    _main(argv, binding_type=OpenWAMPiperBinding, builder=build_server, verifier=verify_checkpoint_directory)


def _main(argv, *, binding_type, builder, verifier, default_port=8848):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-manifest")
    parser.add_argument("--binding", required=True)
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    data = json.loads(Path(args.binding).read_text())
    data["camera_names"] = tuple(data["camera_names"])
    binding = binding_type(**data)
    binding.validate()
    if args.verify_only:
        if args.source_manifest:
            from .openwam_piper_source import verify_source_directory
            verify_source_directory(args.source_root, args.source_manifest)
        else:
            verify_source_checkout(args.source_root)
        print(_json(verifier(args.checkpoint)))
        return
    runtime = builder(checkpoint=args.checkpoint, source_root=args.source_root, binding=binding,
                           source_manifest=args.source_manifest)
    from websockets.sync.server import serve

    try:
        with serve(runtime.serve_connection, "127.0.0.1", args.port, max_size=MAX_MESSAGE_BYTES,
                   compression=None) as listener:
            listener.serve_forever()
    finally:
        runtime.backend.shutdown()


if __name__ == "__main__":
    main()
