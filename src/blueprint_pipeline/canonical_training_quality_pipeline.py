"""Load-bearing canonical quality chain for premium/native PTDP exports.

The chain is intentionally provider-replaceable and fail closed.  Production
embedding/caption backends are fixed argv JSON subprocesses with immutable
model revisions; missing providers still produce a blocked, signed-lineage
manifest rather than silently falling back to fixture heuristics.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .action_normalization import build_action_normalization_from_trace
from .clip_curation_stage import load_clip_records, run_clip_curation_stage
from .common import ensure_dir, utc_now_iso, write_json
from .grounded_clip_caption_stage import run_grounded_clip_caption_stage
from .semantic_dedup_stage import SemanticDedupConfig, run_semantic_dedup_stage

CANONICAL_PIPELINE_SCHEMA_VERSION = "blueprint.canonical_training_quality_pipeline.v1"
CANONICAL_PIPELINE_SIGNATURE_DOMAIN = b"blueprint.canonical-training-quality.v1\0"
SIGNING_KEY_FILE_ENV = "BLUEPRINT_PTDP_SIGNING_KEY_FILE"
SIGNING_KEY_ID_ENV = "BLUEPRINT_PTDP_SIGNING_KEY_ID"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return sorted({str(item).strip() for item in value if str(item).strip()})


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _immutable_revision(value: Any) -> str:
    revision = str(value or "").strip().lower()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("provider_revision_must_be_40_hex_commit")
    return revision


def _command_argv(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("provider_command_argv_must_be_array")
    argv = tuple(str(item) for item in value)
    if not argv or any(not item for item in argv):
        raise ValueError("provider_command_argv_must_be_nonempty")
    return argv


def _run_json_command(
    *, argv: Sequence[str], payload: Mapping[str, Any], timeout_seconds: int
) -> dict[str, Any]:
    completed = subprocess.run(
        list(argv),
        input=json.dumps(dict(payload), sort_keys=True, separators=(",", ":")),
        text=True,
        capture_output=True,
        check=False,
        shell=False,
        timeout=max(1, int(timeout_seconds)),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"provider_command_failed:{completed.returncode}")
    if len(completed.stdout.encode("utf-8")) > 16 * 1024 * 1024:
        raise RuntimeError("provider_command_response_too_large")
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("provider_command_response_not_json") from exc
    if not isinstance(response, Mapping):
        raise RuntimeError("provider_command_response_not_mapping")
    return dict(response)


class CommandEmbeddingProvider:
    """Fixed-argv SigLIP/DINOv3 adapter with immutable model identity."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.name = str(config.get("name") or "").strip().lower()
        if self.name not in {"siglip", "dinov3"}:
            raise ValueError("embedding_provider_name_not_production_approved")
        self.version = str(config.get("version") or "").strip()
        self.model_id = str(config.get("model_id") or "").strip()
        self.revision = _immutable_revision(config.get("revision"))
        self.command_argv = _command_argv(config.get("command_argv"))
        self.timeout_seconds = int(config.get("timeout_seconds") or 120)
        if not self.version or not self.model_id:
            raise ValueError("embedding_provider_identity_incomplete")
        self.production_ready = True

    def embed_image(self, gray: np.ndarray) -> np.ndarray:
        image = np.asarray(gray, dtype=np.uint8)
        if image.ndim != 2 or image.size == 0:
            raise ValueError("embedding_input_must_be_nonempty_grayscale")
        response = _run_json_command(
            argv=self.command_argv,
            timeout_seconds=self.timeout_seconds,
            payload={
                "schema_version": "blueprint.embedding_request.v1",
                "model_id": self.model_id,
                "revision": self.revision,
                "dtype": "uint8",
                "shape": list(image.shape),
                "pixels_base64": base64.b64encode(image.tobytes(order="C")).decode("ascii"),
                "pixels_sha256": hashlib.sha256(image.tobytes(order="C")).hexdigest(),
            },
        )
        embedding = np.asarray(response.get("embedding"), dtype=np.float64).reshape(-1)
        if not embedding.size or not np.isfinite(embedding).all():
            raise RuntimeError("embedding_provider_returned_invalid_vector")
        return embedding


class CommandCaptionProvider:
    """Fixed-argv grounded-caption adapter with immutable model identity."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.name = str(config.get("name") or "").strip()
        self.version = str(config.get("version") or "").strip()
        self.model_id = str(config.get("model_id") or "").strip()
        self.revision = _immutable_revision(config.get("revision"))
        self.command_argv = _command_argv(config.get("command_argv"))
        self.timeout_seconds = int(config.get("timeout_seconds") or 120)
        if not self.name or not self.version or not self.model_id:
            raise ValueError("caption_provider_identity_incomplete")
        self.production_ready = True

    def caption_clip(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return _run_json_command(
            argv=self.command_argv,
            timeout_seconds=self.timeout_seconds,
            payload=request,
        )


def _providers_from_config(
    config: Mapping[str, Any] | None,
) -> tuple[CommandEmbeddingProvider | None, CommandCaptionProvider | None, list[str]]:
    payload = _mapping(config)
    blockers: list[str] = []
    embedding = None
    caption = None
    try:
        embedding = CommandEmbeddingProvider(_mapping(payload.get("embedding_provider")))
    except (TypeError, ValueError) as exc:
        blockers.append(f"canonical_embedding_provider_config_invalid:{exc}")
    try:
        caption = CommandCaptionProvider(_mapping(payload.get("caption_provider")))
    except (TypeError, ValueError) as exc:
        blockers.append(f"canonical_caption_provider_config_invalid:{exc}")
    return embedding, caption, blockers


def _sign_chain_digest(chain_digest: str) -> tuple[dict[str, Any], list[str]]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    configured = str(os.environ.get(SIGNING_KEY_FILE_ENV) or "").strip()
    if not configured:
        return {}, ["canonical_pipeline_signing_key_not_configured"]
    path = Path(configured).expanduser()
    try:
        stat = path.stat()
    except OSError:
        return {}, ["canonical_pipeline_signing_key_unreadable"]
    if path.is_symlink() or not path.is_file() or stat.st_mode & 0o077:
        return {}, ["canonical_pipeline_signing_key_not_private_regular_file"]
    try:
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    except (OSError, TypeError, ValueError):
        return {}, ["canonical_pipeline_signing_key_invalid"]
    if not isinstance(key, Ed25519PrivateKey):
        return {}, ["canonical_pipeline_signing_key_not_ed25519"]
    message = CANONICAL_PIPELINE_SIGNATURE_DOMAIN + bytes.fromhex(chain_digest)
    signature = key.sign(message)
    public_raw = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    key.public_key().verify(signature, message)
    public_sha = hashlib.sha256(public_raw).hexdigest()
    return (
        {
            "algorithm": "Ed25519",
            "domain": "blueprint.canonical-training-quality.v1",
            "key_id": str(os.environ.get(SIGNING_KEY_ID_ENV) or "").strip()
            or f"sha256:{public_sha}",
            "public_key_encoding": "raw_base64",
            "public_key": base64.b64encode(public_raw).decode("ascii"),
            "public_key_sha256": public_sha,
            "signature_encoding": "base64",
            "signature": base64.b64encode(signature).decode("ascii"),
            "verified_at_write": True,
        },
        [],
    )


def run_canonical_training_quality_pipeline(
    *,
    bundle_dir: str | Path,
    action_trace_path: str | Path,
    action_space: Mapping[str, Any] | None,
    provider_config: Mapping[str, Any] | None = None,
    embedding_provider: Any | None = None,
    caption_provider: Any | None = None,
) -> dict[str, Any]:
    """Run every canonical stage in order and emit one signed chain manifest."""

    root = Path(bundle_dir).expanduser().resolve()
    canonical_dir = root / "derived" / "canonical_training_quality"
    ensure_dir(canonical_dir)
    manifest_path = canonical_dir / "canonical_training_quality_pipeline.json"
    configured_embedding, configured_caption, blockers = _providers_from_config(
        provider_config
    )
    embedding = embedding_provider or configured_embedding
    caption = caption_provider or configured_caption
    if embedding_provider is not None:
        blockers = [
            blocker
            for blocker in blockers
            if not blocker.startswith("canonical_embedding_provider_config_invalid:")
        ]
    if caption_provider is not None:
        blockers = [
            blocker
            for blocker in blockers
            if not blocker.startswith("canonical_caption_provider_config_invalid:")
        ]

    curation_path = root / "derived" / "clip_curation" / "clip_curation_manifest.json"
    dedup_path = root / "derived" / "semantic_dedup" / "semantic_dedup_manifest.json"
    caption_path = (
        root
        / "derived"
        / "grounded_clip_captions"
        / "grounded_clip_caption_manifest.json"
    )
    action_path = root / "action_validation_manifest.json"
    trace_path = Path(action_trace_path).expanduser().resolve()

    curation: dict[str, Any] = {}
    dedup: dict[str, Any] = {}
    captions: dict[str, Any] = {}
    action: dict[str, Any] = {}
    try:
        run_clip_curation_stage(bundle_dir=root)
        curation = _read_mapping(curation_path)
    except Exception as exc:  # stage boundary must become a durable blocker
        blockers.append(f"canonical_clip_curation_failed:{type(exc).__name__}")

    curation_ids = _string_list(curation.get("accepted_clip_ids"))
    if not curation_ids:
        blockers.append("canonical_clip_curation_no_accepted_clips")
    if curation_ids:
        try:
            run_semantic_dedup_stage(
                bundle_dir=root,
                provider=embedding,
                config=SemanticDedupConfig(production_mode=True),
                accepted_clip_ids=curation_ids,
            )
            dedup = _read_mapping(dedup_path)
        except Exception as exc:
            blockers.append(f"canonical_semantic_dedup_failed:{type(exc).__name__}")
    if dedup.get("production_status") != "passed":
        blockers.extend(
            f"canonical_semantic_dedup:{item}"
            for item in _string_list(dedup.get("production_blockers"))
        )
        if not _string_list(dedup.get("production_blockers")):
            blockers.append("canonical_semantic_dedup_not_passed")

    if curation and dedup:
        try:
            captions = run_grounded_clip_caption_stage(
                bundle_dir=root,
                curation_manifest_path=curation_path,
                dedup_manifest_path=dedup_path,
                provider=caption,
            )
        except Exception as exc:
            blockers.append(f"canonical_grounded_caption_failed:{type(exc).__name__}")
    if captions.get("status") != "passed":
        blockers.extend(
            f"canonical_grounded_caption:{item}"
            for item in _string_list(captions.get("blockers"))
        )
        if not _string_list(captions.get("blockers")):
            blockers.append("canonical_grounded_caption_not_passed")

    try:
        action = build_action_normalization_from_trace(
            output_dir=root,
            trace=_read_mapping(trace_path),
            source_trace_path=trace_path,
            consumed_by="canonical_training_quality_pipeline",
            action_space=action_space,
        )
    except Exception as exc:
        blockers.append(f"canonical_action_normalization_failed:{type(exc).__name__}")
    if action.get("status") != "validated":
        blockers.extend(
            f"canonical_action_normalization:{item}"
            for item in _string_list(action.get("blockers"))
        )
        if not _string_list(action.get("blockers")):
            blockers.append("canonical_action_normalization_not_validated")

    accepted_clip_ids = _string_list(captions.get("accepted_clip_ids"))
    accepted_episode_ids = sorted(
        str(episode_id)
        for episode_id, result in _mapping(action.get("episode_results")).items()
        if _mapping(result).get("valid") is True
    )
    clip_records: dict[str, dict[str, Any]] = {}
    try:
        for index, clip in enumerate(load_clip_records(root)):
            clip_id = str(
                clip.get("clip_id") or clip.get("id") or f"clip_{index:06d}"
            ).strip()
            clip_records[clip_id] = dict(clip)
    except Exception:
        blockers.append("canonical_clip_manifest_unreadable")
    accepted_attempt_ids: list[str] = []
    for clip_id in accepted_clip_ids:
        attempt_id = str(clip_records.get(clip_id, {}).get("attempt_id") or "").strip()
        if not attempt_id:
            blockers.append(f"canonical_clip_missing_attempt_id:{clip_id}")
        else:
            accepted_attempt_ids.append(attempt_id)
    accepted_attempt_ids = sorted(set(accepted_attempt_ids))
    if set(accepted_attempt_ids) - set(accepted_episode_ids):
        blockers.append("canonical_clip_attempt_not_action_validated")

    stage_paths = {
        "clip_curation": curation_path,
        "semantic_dedup": dedup_path,
        "grounded_clip_caption": caption_path,
        "action_normalization": action_path,
    }
    stage_artifacts = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in stage_paths.items()
    }
    if any(not _mapping(record).get("sha256") for record in stage_artifacts.values()):
        blockers.append("canonical_stage_artifact_missing")
    chain_content = {
        "schema_version": CANONICAL_PIPELINE_SCHEMA_VERSION,
        "clip_manifest_sha256": _sha256_file(root / "clips_manifest.json"),
        "action_trace_sha256": _sha256_file(trace_path),
        "stage_artifact_sha256": {
            name: _mapping(record).get("sha256")
            for name, record in stage_artifacts.items()
        },
        "accepted_clip_ids": accepted_clip_ids,
        "accepted_attempt_ids": accepted_attempt_ids,
    }
    chain_digest = hashlib.sha256(
        json.dumps(chain_content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    signature, signing_blockers = _sign_chain_digest(chain_digest)
    blockers.extend(signing_blockers)
    blockers = sorted(set(blockers))
    manifest = {
        "schema_version": CANONICAL_PIPELINE_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "passed" if not blockers else "blocked",
        "stage_order": [
            "clip_curation",
            "semantic_dedup",
            "grounded_clip_caption",
            "action_alignment_and_normalization",
        ],
        "stage_artifacts": stage_artifacts,
        "chain_content": chain_content,
        "chain_digest_sha256": chain_digest,
        "chain_signature": signature or None,
        "accepted_clip_ids": accepted_clip_ids if not blockers else [],
        "accepted_attempt_ids": accepted_attempt_ids if not blockers else [],
        "rejected_clip_ids": sorted(set(clip_records) - set(accepted_clip_ids)),
        "blockers": blockers,
        "claim_boundary": {
            "premium_quality_eligible": not blockers,
            "native_training_export_eligible": not blockers,
            "self_attested_metadata_is_canonical_stage_proof": False,
            "rejected_clips_exported": False,
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def run_canonical_training_quality_from_request(
    *, job_dir: str | Path, request: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(job_dir).expanduser().resolve()
    trace_path = (
        root / "normalized_attempt_trace.json"
        if (root / "normalized_attempt_trace.json").is_file()
        else root / "policy_execution_trace.json"
    )
    policy_package = _mapping(request.get("policy_package") or request.get("policyPackage"))
    action_space = _mapping(
        request.get("action_space")
        or request.get("actionSpace")
        or policy_package.get("action_space")
        or policy_package.get("actionSpace")
    )
    provider_config = _mapping(
        request.get("canonical_training_quality")
        or request.get("canonicalTrainingQuality")
    )
    return run_canonical_training_quality_pipeline(
        bundle_dir=root,
        action_trace_path=trace_path,
        action_space=action_space,
        provider_config=provider_config,
    )


__all__ = [
    "CANONICAL_PIPELINE_SCHEMA_VERSION",
    "CommandCaptionProvider",
    "CommandEmbeddingProvider",
    "run_canonical_training_quality_from_request",
    "run_canonical_training_quality_pipeline",
]
