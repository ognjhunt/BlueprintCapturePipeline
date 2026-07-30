"""Identity-bound server wrapper for NVIDIA's frozen Policy-DROID endpoint.

NVIDIA's public RoboLab server intentionally sends empty WebSocket metadata.
Blueprint loads the unmodified official policy service but serves it with the
verified source/model/snapshot identity required by scientific evidence.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .common import write_json
from .cosmos_edge_droid_policy_runtime import (
    NATIVE_ACTION_CHUNK_ROWS,
    CosmosEdgeDroidPolicySpec,
    verify_local_policy_snapshot,
)
from .droid_policy_bridge import validate_droid_action_chunk


class NativeActionShapeGuard:
    """Retain NVIDIA service behavior while failing closed on contract drift."""

    def __init__(self, service: Any) -> None:
        self._service = service

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        raw = self._service.infer(observation)
        if not isinstance(raw, Mapping):
            raise ValueError("cosmos_edge_policy_native_response_not_object")
        action = raw.get("action", raw.get("actions"))
        blockers = validate_droid_action_chunk(action, expected_rows=NATIVE_ACTION_CHUNK_ROWS)
        if blockers:
            raise ValueError(f"cosmos_edge_policy_native_action_invalid:{blockers[0]}")
        result = dict(raw)
        result["action"] = np.asarray(action, dtype=np.float64)
        result.pop("actions", None)
        return result


def _disable_policy_guardrails(setup_args: Any) -> Any:
    """Disable the optional post-generation guardrail dependency.

    The pinned NVIDIA RoboLab service inherits the Cosmos inference default of
    loading ``nvidia/Cosmos-Guardrail1`` before the policy can return structured
    robot actions. NVIDIA's setup model exposes this as a supported setting,
    and NVIDIA's export verifier uses the same setting because the guardrail is
    an orthogonal Hugging Face dependency. The pinned inference source applies
    it to prompt/video acceptance after generation, not to policy weights,
    conditioning, or native action shape. Blueprint's own action validation,
    uncertainty, and abstention gates remain in force.
    """

    if not isinstance(getattr(setup_args, "guardrails", None), bool):
        raise TypeError("cosmos_edge_policy_setup_args_guardrails_field_unavailable")
    if not hasattr(setup_args, "model_copy"):
        raise TypeError("cosmos_edge_policy_setup_args_model_copy_unavailable")
    updated = setup_args.model_copy(update={"guardrails": False})
    if getattr(updated, "guardrails", None) is not False:
        raise ValueError("cosmos_edge_policy_guardrails_disable_failed")
    return updated


def serve_identity_bound_policy(
    *,
    checkpoint_path: str | Path,
    snapshot_manifest_path: str | Path,
    host: str,
    port: int,
    output_dir: str | Path,
    service_factory: Callable[..., Any] | None = None,
    server_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Verify the full snapshot, load NVIDIA's service, and serve exact metadata."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    manifest_path = Path(snapshot_manifest_path).expanduser().resolve()
    if not host.strip() or not 1 <= int(port) <= 65535:
        raise ValueError("cosmos_edge_policy_server_endpoint_invalid")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = CosmosEdgeDroidPolicySpec(
        snapshot_manifest_sha256=str(manifest.get("manifest_sha256") or "")
    )
    verification = verify_local_policy_snapshot(
        spec=spec,
        snapshot_dir=checkpoint,
        snapshot_manifest_path=manifest_path,
    )
    if service_factory is None or server_factory is None:
        try:
            from cosmos_framework.scripts.action_policy_server_robolab import (
                RobolabPolicyService,
                RobolabServerArgs,
                _load_openpi_websocket_policy_server,
            )
        except ImportError as exc:  # pragma: no cover - GPU runtime dependency
            raise RuntimeError("pinned_cosmos_framework_policy_server_unavailable") from exc
        args = RobolabServerArgs(
            checkpoint_path=str(checkpoint),
            hf_revision=spec.model_revision,
            port=int(port),
            host=host,
            output_dir=Path(output_dir).expanduser().resolve() / "nvidia_policy_runtime",
            action_chunk_size=NATIVE_ACTION_CHUNK_ROWS,
            action_dim=8,
            conditioning_fps=15.0,
            action_space="joint_pos",
            use_state=True,
            history_length=1,
            deterministic_seed=True,
            seed=0,
            guidance=3.0,
            num_steps=4,
            shift=5.0,
            format_prompt_as_json=True,
        )

        class BlueprintRobolabPolicyService(RobolabPolicyService):
            def _build_setup_args(self, service_args: Any) -> Any:
                return _disable_policy_guardrails(super()._build_setup_args(service_args))

        def _official_service_factory() -> Any:
            return BlueprintRobolabPolicyService(args)

        server_cls = _load_openpi_websocket_policy_server()

        def _official_server_factory(**kwargs: Any) -> Any:
            return server_cls(**kwargs)

        service_factory = _official_service_factory
        server_factory = _official_server_factory
    service = NativeActionShapeGuard(service_factory())
    metadata = {**spec.server_metadata(), **verification}
    startup = {
        "schema_version": "cosmos_edge_droid_policy_server_startup.v1",
        "status": "model_loaded_ready_to_serve",
        "host": host,
        "port": int(port),
        "metadata": metadata,
        "native_action_shape": [NATIVE_ACTION_CHUNK_ROWS, 8],
        "wam_prefix_adapter_runs_client_side": True,
        "nvidia_guardrails_enabled": False,
        "guardrail_mode": "disabled_source_supported_post_generation_filter",
        "policy_checkpoint_or_action_contract_modified_by_guardrail_override": False,
        "blueprint_action_and_abstention_gates_remain_enabled": True,
        "raw_credentials_written": False,
    }
    startup_path = Path(output_dir).expanduser().resolve() / "policy_server_startup.json"
    startup_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(startup_path, startup)
    server = server_factory(policy=service, host=host, port=int(port), metadata=metadata)
    server.serve_forever()
    return startup


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--snapshot-manifest", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    serve_identity_bound_policy(
        checkpoint_path=args.checkpoint_path,
        snapshot_manifest_path=args.snapshot_manifest,
        host=args.host,
        port=args.port,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["NativeActionShapeGuard", "serve_identity_bound_policy"]
