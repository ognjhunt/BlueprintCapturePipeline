"""C2PA edge stamping for customer-facing package media — sidecar-only.

The internal Blueprint ledger stays authoritative for rights, capture truth,
evaluation meaning, and claim eligibility. A C2PA manifest is only a signed
interoperability wrapper proving what Blueprint asserted about the exported
bytes. Package media objects are content-addressed (filename == sha256 of the
bytes), so this module never mutates asset bytes: manifests are written as
adjacent ``.c2pa`` sidecar files, and a stamp is claimed only after a
round-trip verification and a byte-identity check on the asset.

Fail-closed contract: missing tool, cert, or key downgrades to an explicit
``unavailable`` record — exports proceed unstamped and the absence is visible
in the export manifest and buyer readout, never silently and never as a false
"stamped" claim. Signing is server-side via a pinned external ``c2patool``
binary (configured by env), which is not a Python dependency of this package.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCHEMA_VERSION = "c2pa_edge_stamping.v1"
LEDGER_REF_ASSERTION_LABEL = "com.blueprint.ledger_ref"
LEDGER_REF_ASSERTION_SCHEMA_VERSION = "blueprint.c2pa_ledger_ref.v1"

C2PA_ENABLED_ENV = "BLUEPRINT_PTDP_C2PA_STAMPING"
C2PA_TOOL_BIN_ENV = "BLUEPRINT_PTDP_C2PATOOL_BIN"
C2PA_SIGN_CERT_FILE_ENV = "BLUEPRINT_PTDP_C2PA_SIGN_CERT_FILE"
C2PA_SIGN_KEY_FILE_ENV = "BLUEPRINT_PTDP_C2PA_SIGN_KEY_FILE"
C2PA_VERIFICATION_URI_ENV = "BLUEPRINT_PTDP_C2PA_VERIFICATION_URI"
C2PA_3D_EXPERIMENTAL_ENV = "BLUEPRINT_PTDP_C2PA_3D_EXPERIMENTAL"

RECORD_RELATIVE_PATH = "exports/provenance/c2pa_stamping_record.json"

# Formats with C2PA embedding/sidecar bindings that c2patool supports today.
# The package also emits .avi/.mkv/.webm clips; those are recorded as
# unsupported_format rather than silently skipped (C2PA has no binding).
STAMPABLE_MEDIA_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".png", ".jpg", ".jpeg"})

_DIGEST_PATTERN = re.compile(r"^(sha256:)?[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")
_URI_PATTERN = re.compile(r"^https://[^\s]{1,512}$")

_ALLOWED_LEDGER_REF_KEYS = frozenset(
    {
        "scene_id",
        "capture_id",
        "consent_evidence_digest",
        "signed_chain_manifest_sha256",
        "holdout_split_sha256",
        "success_claim_ledger_digest",
        "verification_uri",
    }
)

_SUBPROCESS_TIMEOUT_SECONDS = 120


class C2paStampingError(RuntimeError):
    """Raised for stamping-configuration misuse (never for tool absence)."""


class C2paAssertionContentError(ValueError):
    """Raised when a ledger-ref assertion would carry non-digest content."""


Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


def _default_runner(command: list[str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(  # noqa: S603 - pinned binary from explicit env config
        command,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )


def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def _env_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_ledger_ref_value(key: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise C2paAssertionContentError(f"c2pa_ledger_ref_value_empty:{key}")
    if key == "verification_uri":
        if not _URI_PATTERN.match(text):
            raise C2paAssertionContentError(f"c2pa_ledger_ref_uri_invalid:{key}")
        return text
    if key.endswith(("_digest", "_sha256")):
        if not _DIGEST_PATTERN.match(text):
            raise C2paAssertionContentError(f"c2pa_ledger_ref_digest_invalid:{key}")
        return text
    if not _IDENTIFIER_PATTERN.match(text) or " " in text:
        raise C2paAssertionContentError(f"c2pa_ledger_ref_identifier_invalid:{key}")
    return text


def build_ledger_ref_assertion(
    *,
    artifact_relative_path: str,
    artifact_sha256: str,
    ledger_refs: Mapping[str, Any],
) -> dict[str, Any]:
    """Digest-and-identifier-only assertion payload; rights content is refused.

    The assertion references the internal ledger by digest — it never carries
    the rights record, consent scope text, or any other free-form content.
    """

    unknown = sorted(set(ledger_refs) - _ALLOWED_LEDGER_REF_KEYS)
    if unknown:
        raise C2paAssertionContentError(
            "c2pa_ledger_ref_key_not_allowed:" + ",".join(unknown)
        )
    if not _DIGEST_PATTERN.match(str(artifact_sha256 or "")):
        raise C2paAssertionContentError("c2pa_ledger_ref_digest_invalid:artifact_sha256")
    ledger: dict[str, str] = {}
    for key in ("consent_evidence_digest", "signed_chain_manifest_sha256",
                "holdout_split_sha256", "success_claim_ledger_digest"):
        if ledger_refs.get(key):
            ledger[key] = _validate_ledger_ref_value(key, ledger_refs[key])
    return {
        "schema_version": LEDGER_REF_ASSERTION_SCHEMA_VERSION,
        "artifact": {
            "relative_path": str(artifact_relative_path),
            "sha256": str(artifact_sha256),
        },
        "ledger": ledger,
        "scene_id": _validate_ledger_ref_value("scene_id", ledger_refs.get("scene_id")),
        "capture_id": _validate_ledger_ref_value(
            "capture_id", ledger_refs.get("capture_id")
        ),
        "verification_uri": _validate_ledger_ref_value(
            "verification_uri", ledger_refs.get("verification_uri")
        )
        if ledger_refs.get("verification_uri")
        else None,
        "internal_ledger_authoritative": True,
    }


def resolve_stamping_config(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Fail-closed config resolution; every gap is an explicit blocker."""

    mapping = _env(env)
    enabled = _env_truthy(mapping.get(C2PA_ENABLED_ENV))
    blockers: list[str] = []
    tool_bin = str(mapping.get(C2PA_TOOL_BIN_ENV) or "").strip()
    cert_file = str(mapping.get(C2PA_SIGN_CERT_FILE_ENV) or "").strip()
    key_file = str(mapping.get(C2PA_SIGN_KEY_FILE_ENV) or "").strip()
    if enabled:
        if not tool_bin or not Path(tool_bin).is_file():
            blockers.append("c2pa_tool_bin_missing")
        if not cert_file or not Path(cert_file).is_file():
            blockers.append("c2pa_sign_cert_file_missing")
        if not key_file or not Path(key_file).is_file():
            blockers.append("c2pa_sign_key_file_missing")
    return {
        "enabled": enabled,
        "tool_bin": tool_bin or None,
        "sign_cert_file": cert_file or None,
        "sign_key_file": key_file or None,
        "verification_uri": str(mapping.get(C2PA_VERIFICATION_URI_ENV) or "").strip()
        or None,
        "blockers": blockers,
    }


def _manifest_definition(
    *,
    assertion: Mapping[str, Any],
    title: str,
    sign_cert_file: str,
    sign_key_file: str,
) -> dict[str, Any]:
    return {
        "claim_generator_info": [
            {"name": "blueprint-capture-pipeline", "version": SCHEMA_VERSION}
        ],
        "title": title,
        "alg": "es256",
        "sign_cert": sign_cert_file,
        "private_key": sign_key_file,
        "assertions": [
            {"label": LEDGER_REF_ASSERTION_LABEL, "data": dict(assertion)}
        ],
    }


def _stamp_one_media_file(
    *,
    package_dir: Path,
    relative_path: str,
    ledger_refs: Mapping[str, Any],
    config: Mapping[str, Any],
    runner: Runner,
) -> dict[str, Any]:
    asset_path = package_dir / relative_path
    record: dict[str, Any] = {
        "relative_path": relative_path,
        "status": "failed",
        "sidecar_relative_path": None,
        "asset_sha256": None,
        "blockers": [],
    }
    if not asset_path.is_file():
        record["blockers"].append("c2pa_asset_missing")
        return record
    if asset_path.suffix.lower() not in STAMPABLE_MEDIA_SUFFIXES:
        record["status"] = "unsupported_format"
        record["blockers"].append(
            f"c2pa_format_binding_unavailable:{asset_path.suffix.lower()}"
        )
        return record

    asset_sha256 = _sha256_file(asset_path)
    record["asset_sha256"] = asset_sha256
    assertion = build_ledger_ref_assertion(
        artifact_relative_path=relative_path,
        artifact_sha256=asset_sha256,
        ledger_refs=ledger_refs,
    )
    with tempfile.TemporaryDirectory(prefix="c2pa-stamp-") as workdir_str:
        workdir = Path(workdir_str)
        workdir.chmod(0o700)
        source = workdir / asset_path.name
        shutil.copyfile(asset_path, source)
        definition_path = workdir / "manifest_definition.json"
        definition_path.write_text(
            json.dumps(
                _manifest_definition(
                    assertion=assertion,
                    title=asset_path.name,
                    sign_cert_file=str(config["sign_cert_file"]),
                    sign_key_file=str(config["sign_key_file"]),
                ),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        output = workdir / "out" / asset_path.name
        output.parent.mkdir()
        sign = runner(
            [
                str(config["tool_bin"]),
                str(source),
                "-m",
                str(definition_path),
                "--sidecar",
                "-o",
                str(output),
            ]
        )
        if sign.returncode != 0:
            record["blockers"].append("c2pa_sign_invocation_failed")
            return record
        sidecar = output.with_suffix(".c2pa")
        if not output.is_file() or not sidecar.is_file():
            record["blockers"].append("c2pa_sidecar_not_produced")
            return record
        if _sha256_file(output) != asset_sha256:
            # Sidecar mode must never rewrite the asset: the object store is
            # content-addressed, so changed bytes would break every digest
            # upstream of this call. Refuse the stamp entirely.
            record["blockers"].append("c2pa_output_asset_bytes_changed")
            return record
        verify = runner([str(config["tool_bin"]), str(output)])
        if verify.returncode != 0 or LEDGER_REF_ASSERTION_LABEL not in str(
            verify.stdout
        ):
            record["blockers"].append("c2pa_verification_round_trip_failed")
            return record
        installed_sidecar = asset_path.with_suffix(".c2pa")
        shutil.copyfile(sidecar, installed_sidecar)
        record["status"] = "stamped"
        record["sidecar_relative_path"] = str(installed_sidecar.relative_to(package_dir))
        record["manifest_store_sha256"] = _sha256_file(installed_sidecar)
    return record


def apply_edge_stamping(
    *,
    package_dir: Path,
    media_relative_paths: Sequence[str],
    ledger_refs: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Stamp package media with sidecar C2PA manifests; fail closed on gaps.

    Returns (and, when stamping was attempted, persists) a
    ``c2pa_edge_stamping.v1`` record. Statuses: ``disabled`` (flag off),
    ``unavailable`` (flag on, tool/signer missing), ``stamped`` (every
    supported file stamped and verified), ``partial`` (some stamped),
    ``failed`` (attempted, none stamped).
    """

    package_dir = Path(package_dir)
    config = resolve_stamping_config(env)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "disabled",
        "sidecar_only": True,
        "internal_ledger_authoritative": True,
        "total_media_count": len(media_relative_paths),
        "stamped_count": 0,
        "blockers": [],
        "files": [],
        "record_path": None,
        "tool_bin": config["tool_bin"],
    }
    if not config["enabled"]:
        return record
    if config["blockers"]:
        record["status"] = "unavailable"
        record["blockers"] = list(config["blockers"])
        return _persist_record(package_dir, record)

    active_runner = runner or _default_runner
    files: list[dict[str, Any]] = []
    for relative_path in media_relative_paths:
        files.append(
            _stamp_one_media_file(
                package_dir=package_dir,
                relative_path=str(relative_path),
                ledger_refs=ledger_refs,
                config=config,
                runner=active_runner,
            )
        )
    record["files"] = files
    stamped = sum(1 for row in files if row["status"] == "stamped")
    supported = [row for row in files if row["status"] != "unsupported_format"]
    record["stamped_count"] = stamped
    if supported and stamped == len(supported):
        record["status"] = "stamped"
    elif stamped:
        record["status"] = "partial"
    else:
        record["status"] = "failed"
    return _persist_record(package_dir, record)


def _ptdp_media_relative_paths(output_dir: Path) -> list[str]:
    roots = (
        output_dir / "exports" / "video_bundle" / "objects",
        output_dir / "exports" / "lerobot_v3" / "videos",
        output_dir / "exports" / "gr00t_lerobot" / "videos",
    )
    media: list[str] = []
    package_media_suffixes = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in package_media_suffixes:
                media.append(str(path.relative_to(output_dir)))
    return media


def apply_ptdp_edge_stamping(
    output_dir: Path,
    manifest: dict[str, Any],
    *,
    env: Mapping[str, str] | None = None,
) -> None:
    """Apply sidecar-only C2PA stamping at the PTDP export edge.

    This runs after final media bytes exist and before package integrity
    artifacts are produced. It never mutates media bytes and never blocks an
    export: failures are represented explicitly in the manifest.
    """

    context_value = manifest.get("context")
    context = context_value if isinstance(context_value, Mapping) else {}
    ledger_refs: dict[str, Any] = {
        "scene_id": str(context.get("scene_id") or "unknown"),
        "capture_id": str(context.get("capture_id") or "unknown"),
    }
    for ref_key, artifact_name in (
        ("consent_evidence_digest", "consent_evidence.json"),
        ("signed_chain_manifest_sha256", "canonical_training_quality_pipeline.json"),
    ):
        artifact_path = output_dir / artifact_name
        if artifact_path.is_file():
            ledger_refs[ref_key] = f"sha256:{_sha256_file(artifact_path)}"
    holdout_path = output_dir / "holdout_split.json"
    if holdout_path.is_file():
        try:
            holdout_sha = str(
                json.loads(holdout_path.read_text(encoding="utf-8")).get("split_sha256")
                or ""
            )
        except (OSError, json.JSONDecodeError):
            holdout_sha = ""
        if holdout_sha:
            ledger_refs["holdout_split_sha256"] = holdout_sha
    try:
        record = apply_edge_stamping(
            package_dir=output_dir,
            media_relative_paths=_ptdp_media_relative_paths(output_dir),
            ledger_refs=ledger_refs,
            env=env,
        )
        summary = {
            key: record.get(key)
            for key in (
                "schema_version",
                "status",
                "sidecar_only",
                "internal_ledger_authoritative",
                "total_media_count",
                "stamped_count",
                "blockers",
                "record_path",
            )
        }
    except Exception as exc:  # noqa: BLE001 - stamping must never block the export
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "sidecar_only": True,
            "internal_ledger_authoritative": True,
            "total_media_count": 0,
            "stamped_count": 0,
            "blockers": [f"c2pa_stamping_exception:{type(exc).__name__}"],
            "record_path": None,
        }
    manifest["c2pa_edge_stamping"] = summary


def _persist_record(package_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    record_path = package_dir / RECORD_RELATIVE_PATH
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record["record_path"] = RECORD_RELATIVE_PATH
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def stamp_3d_asset_sidecar(
    *,
    asset_path: Path,
    ledger_refs: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """3D C2PA sidecars are not qualified: no format binding exists for
    glTF/USD/PLY/SPZ, and the data-hash binding, tamper behavior, and
    validator discovery have not been proven against committed fixtures.
    Fail closed regardless of the experimental flag until that prototype
    lands; the Blueprint checksum/provenance sidecars remain the 3D truth.
    """

    del ledger_refs
    experimental_requested = _env_truthy(_env(env).get(C2PA_3D_EXPERIMENTAL_ENV))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked_experimental_not_qualified",
        "asset": str(asset_path),
        "experimental_flag_requested": experimental_requested,
        "blockers": [
            "c2pa_3d_format_binding_unqualified",
            "c2pa_3d_data_hash_binding_not_proven_on_fixtures",
            "c2pa_3d_validator_discovery_not_proven",
        ],
    }


__all__ = [
    "SCHEMA_VERSION",
    "LEDGER_REF_ASSERTION_LABEL",
    "LEDGER_REF_ASSERTION_SCHEMA_VERSION",
    "C2PA_ENABLED_ENV",
    "C2PA_TOOL_BIN_ENV",
    "C2PA_SIGN_CERT_FILE_ENV",
    "C2PA_SIGN_KEY_FILE_ENV",
    "C2PA_VERIFICATION_URI_ENV",
    "C2PA_3D_EXPERIMENTAL_ENV",
    "STAMPABLE_MEDIA_SUFFIXES",
    "RECORD_RELATIVE_PATH",
    "C2paStampingError",
    "C2paAssertionContentError",
    "apply_edge_stamping",
    "apply_ptdp_edge_stamping",
    "build_ledger_ref_assertion",
    "resolve_stamping_config",
    "stamp_3d_asset_sidecar",
]
