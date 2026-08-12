"""Execute the exact uploaded provider entrypoint up to its paid/GPU boundary."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .provider_archive import extract_provider_archive


class ProviderBundleRehearsalError(ValueError):
    """The exact immutable provider bundle did not pass its zero-cost seam."""


def provider_bundle_rehearsal_blockers(
    rehearsal: Any,
    *,
    bundle_sha256: str,
    entrypoint_relative_path: str,
) -> list[str]:
    """Validate that the receipt belongs to these exact bytes and entrypoint."""

    if not isinstance(rehearsal, dict):
        return ["exact_bundle_entrypoint_rehearsal_missing"]
    valid = (
        rehearsal.get("status") == "passed"
        and rehearsal.get("bundle_sha256") == bundle_sha256
        and rehearsal.get("entrypoint_relative_path") == entrypoint_relative_path
        and rehearsal.get("returncode") == 0
        and rehearsal.get("gpu_runtime_started") is False
        and rehearsal.get("paid_inference_performed") is False
        and rehearsal.get("provider_mutations_performed") == 0
    )
    return [] if valid else ["exact_bundle_entrypoint_rehearsal_invalid"]


def rehearse_provider_bundle_entrypoint(
    *,
    bundle_path: str | Path,
    entrypoint_relative_path: str,
    evidence_path: str | Path,
) -> dict[str, Any]:
    """Extract the exact bundle and run its real shell in rehearsal mode."""

    bundle = Path(bundle_path).expanduser().resolve()
    evidence = Path(evidence_path).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="blueprint-provider-rehearsal-") as raw:
        root = (Path(raw) / "bundle").resolve()
        extract_provider_archive(bundle, root)
        entrypoint = (root / entrypoint_relative_path).resolve()
        if root != entrypoint and root not in entrypoint.parents:
            raise ProviderBundleRehearsalError("provider_rehearsal_entrypoint_outside_bundle")
        if not entrypoint.is_file():
            raise ProviderBundleRehearsalError("provider_rehearsal_entrypoint_missing")
        output = Path(raw) / "runtime_output"
        output.mkdir()
        environment = os.environ.copy()
        environment.update(
            {
                "BLUEPRINT_PROVIDER_BUNDLE_REHEARSAL": "1",
                "BLUEPRINT_ADP_CONTENT_AGENTS_OUTPUT_DIR": str(output),
                "BLUEPRINT_ADP_JOINT_AGENT_OUTPUT_DIR": str(output),
                "BLUEPRINT_ADP_GAUSSIAN_EXCISION_OUTPUT_DIR": str(output),
                "BLUEPRINT_ADP_ARENA_OUTPUT_DIR": str(output),
                "BLUEPRINT_PUBLIC_SCENE_AURA_EXACT_RESIDUAL_OUTPUT_DIR": str(output),
            }
        )
        result = subprocess.run(
            ["bash", str(entrypoint)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=120,
        )
        receipt_path = output / "provider_bundle_rehearsal.json"
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProviderBundleRehearsalError(
                "provider_rehearsal_receipt_missing_or_invalid"
            ) from exc
        if (
            result.returncode != 0
            or not isinstance(receipt, dict)
            or receipt.get("status") != "passed"
            or receipt.get("paid_inference_performed") is not False
            or receipt.get("gpu_runtime_started") is not False
            or receipt.get("provider_mutations_performed") != 0
        ):
            raise ProviderBundleRehearsalError("provider_rehearsal_failed")
        final = {
            **receipt,
            "bundle_path": str(bundle),
            "bundle_sha256": "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest(),
            "entrypoint_relative_path": entrypoint_relative_path,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return final


__all__ = [
    "ProviderBundleRehearsalError",
    "provider_bundle_rehearsal_blockers",
    "rehearse_provider_bundle_entrypoint",
]
