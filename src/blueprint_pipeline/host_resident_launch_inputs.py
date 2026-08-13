"""Refuse a paid launch whose inputs live on the machine that authored them.

A bundle receipt records the absolute path of every byte it seals. Those paths
are correct exactly once -- on the workstation that built the bundle. Ship the
receipt to the control plane and each one becomes a claim about a filesystem
that is not there, while the receipt still says ``status: ready``.

Observed on the live control plane on 2026-08-12: the published retained-scene
render profile handed the allocator
``/Users/<author>/workspace/BlueprintValidation/.../adp_..._bundle_receipt.json``
and the run reached the provider boundary and rented a GPU. It resolved only
because a previous session had recreated the author's directory tree on the
droplet by hand. Every existence check passed; nothing about that arrangement
survives a host rebuild, and nothing about it was written down.

So residency is not "the path exists". Two things must hold before a receipt may
report ready:

* the bytes resolve **here**, by digest, not by remembered path; and
* they resolve **under a directory the control plane owns**, so that a
  hand-recreated authoring tree is refused even when its bytes are correct.

Resolution therefore prefers references relative to the receipt's own directory
-- the only form that travels with the bytes -- and falls back to the recorded
absolute path only when it both digest-matches and lies under a control-plane
root.

When no control-plane root exists on this host, the process is not a control
plane (a developer workstation, a CI runner), residency cannot be judged, and
this module does not claim it: digest resolution still applies, the root
restriction does not. That is a deliberate scope limit, not a fallback -- the
deployed host always has ``/var/lib/blueprint``.

Reads retained bytes only; performs no provider mutation.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

RESOLUTION_SCHEMA = "host_resident_bundle_receipt_resolution.v1"

# Directories a Blueprint control-plane host owns. A launch input outside all of
# them belongs to some other machine's filesystem layout.
PRODUCTION_LAUNCH_INPUT_ROOTS = (
    "/etc/blueprint",
    "/opt/blueprint",
    "/var/lib/blueprint",
)
LAUNCH_INPUT_ROOTS_ENV = "BLUEPRINT_TASK_EVALUATION_LAUNCH_INPUT_ROOTS"

# The receipt fields that name bytes the run depends on. ``bundle`` is the
# archive itself; the other two are the documents that authorize and describe
# it, and a receipt that cannot resolve them is not usable evidence either.
_BUNDLE_REFERENCE = "bundle"
_RECORD_REFERENCES = ("execution_authority", "request")
ARTIFIXER3D_BUNDLE_SCHEMA = "public_scene_artifixer3d_bundle.v1"
ARTIFIXER3D_BUNDLE_READY_STATUS = (
    "sealed_rehearsal_passed_no_upload_no_execution"
)
ARTIFIXER3D_LIVE_PIPELINE_MODE = "dual_target_artifixer3d_only"


class HostResidentInputError(ValueError):
    """A launch input cannot be resolved on this host."""


def _launch_receipt_view(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Project a lane-native receipt onto the portable launch contract.

    Most paid-lane receipts already expose ``bundle_path``, ``bundle_sha256``,
    ``implementation_commit``, and ``status: ready`` at the top level.  The
    ArtiFixer3D receipt predates that contract and deliberately keeps its
    scientific status plus bundle record nested.  The allocator must still
    receive those original bytes, so rewriting the retained receipt is not an
    option; this read-only view supplies only the fields the residency and live
    profile layers share.

    Only the paired whole-frame-teacher mode is launch-ready.  A historical
    direct-editor, broad-repair, Aura, or Inpaint-compatible receipt remains in
    its native status and therefore cannot become a selectable live profile.
    """

    view = json.loads(json.dumps(dict(receipt)))
    if receipt.get("schema_version") != ARTIFIXER3D_BUNDLE_SCHEMA:
        return view

    bundle = receipt.get("bundle")
    identity = receipt.get("blueprint_source_identity")
    rehearsal = receipt.get("local_rehearsal")
    if isinstance(bundle, Mapping):
        view["bundle_path"] = bundle.get("path")
        view["bundle_sha256"] = bundle.get("sha256")
        recorded = str(bundle.get("path") or "").strip()
        if recorded:
            view["bundle_relative_path"] = Path(recorded).name
    if isinstance(identity, Mapping):
        view["implementation_commit"] = identity.get("commit")

    view["native_receipt_status"] = receipt.get("status")
    paired_ready = (
        receipt.get("status") == ARTIFIXER3D_BUNDLE_READY_STATUS
        and receipt.get("pipeline_mode") == ARTIFIXER3D_LIVE_PIPELINE_MODE
        and receipt.get("direct_editor_backend") == "none"
        and receipt.get("semantic_editor_only") is False
        and receipt.get("provider_mutations_performed") == 0
        and isinstance(identity, Mapping)
        and identity.get("tracked_files_clean") is True
        and isinstance(rehearsal, Mapping)
        and rehearsal.get("status") == "passed"
        and rehearsal.get("pipeline_mode") == ARTIFIXER3D_LIVE_PIPELINE_MODE
        and rehearsal.get("provider_mutations_performed") == 0
        and rehearsal.get("paid_inference_performed") is False
        and rehearsal.get("gpu_runtime_started") is False
    )
    if paired_ready:
        view["status"] = "ready"
    return view


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def configured_launch_input_roots(
    env: Mapping[str, str] | None = None,
) -> tuple[Path, ...]:
    """Return the directories this host accepts launch inputs from.

    An explicit configuration wins so a non-default deployment can declare its
    own layout. Otherwise the production roots that actually exist here are
    used, which is what makes the check self-enabling on a control plane and
    silent on a workstation.
    """

    source = os.environ if env is None else env
    configured = str(source.get(LAUNCH_INPUT_ROOTS_ENV) or "").strip()
    if configured:
        roots = [
            Path(item).expanduser()
            for item in configured.split(os.pathsep)
            if item.strip()
        ]
    else:
        roots = [Path(item) for item in PRODUCTION_LAUNCH_INPUT_ROOTS]
    resolved: list[Path] = []
    for root in roots:
        try:
            candidate = root.resolve()
        except OSError:
            continue
        if candidate.is_dir() and candidate not in resolved:
            resolved.append(candidate)
    return tuple(resolved)


def path_is_host_resident(path: str | Path, roots: Sequence[Path]) -> bool:
    """Report whether ``path`` lies under a directory the control plane owns.

    With no roots configured this host is not a control plane, so there is no
    ownership question to answer and every path passes.
    """

    if not roots:
        return True
    try:
        candidate = Path(path).expanduser().resolve()
    except OSError:
        return False
    for root in roots:
        if candidate == root or root in candidate.parents:
            return True
    return False


def _reference_records(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {
        _BUNDLE_REFERENCE: {
            "path": receipt.get("bundle_path"),
            "sha256": receipt.get("bundle_sha256"),
            "relative_path": receipt.get("bundle_relative_path"),
        }
    }
    for name in _RECORD_REFERENCES:
        record = receipt.get(name)
        # A record without a digest is reported as an invalid declaration
        # rather than skipped: an unverifiable reference is the failure this
        # module exists to name, not one to pass over quietly.
        if isinstance(record, Mapping):
            references[name] = {
                "path": record.get("path"),
                "sha256": record.get("sha256"),
                "relative_path": record.get("relative_path"),
            }
    return references


def _safe_relative(root: Path, relative: Any) -> Path | None:
    text = str(relative or "").strip()
    if not text or text.startswith("/") or ".." in Path(text).parts:
        return None
    return root / text


def _candidates(receipt_dir: Path, record: Mapping[str, Any]) -> list[Path]:
    """Portable references first, then bundle-resident copies, then the
    recorded path. Order matters: the first digest match wins, and the recorded
    path is the one form that cannot have travelled with the bytes."""

    candidates: list[Path] = []
    relative = _safe_relative(receipt_dir, record.get("relative_path"))
    if relative is not None:
        candidates.append(relative)
    recorded = str(record.get("path") or "").strip()
    if recorded:
        name = Path(recorded).name
        if name:
            candidates.append(receipt_dir / name)
            candidates.append(receipt_dir / "provider_runtime" / name)
        candidates.append(Path(recorded).expanduser())
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _resolve_reference(
    name: str,
    record: Mapping[str, Any],
    *,
    receipt_dir: Path,
    roots: Sequence[Path],
) -> tuple[dict[str, Any] | None, list[str]]:
    expected = str(record.get("sha256") or "").strip()
    if not expected.startswith("sha256:"):
        return None, [f"host_resident_input_declaration_invalid:{name}"]
    outside_root = False
    for candidate in _candidates(receipt_dir, record):
        try:
            if candidate.is_symlink() or not candidate.is_file():
                continue
            if _sha256(candidate) != expected:
                continue
        except OSError:
            continue
        if not path_is_host_resident(candidate, roots):
            # The bytes are right but they are sitting in some other machine's
            # directory layout. Keep looking; report this only if nothing else
            # resolves, so the operator learns which failure they have.
            outside_root = True
            continue
        return (
            {
                "name": name,
                "path": str(candidate.resolve()),
                "sha256": expected,
                "size_bytes": candidate.stat().st_size,
            },
            [],
        )
    if outside_root:
        return None, [f"host_resident_input_outside_control_plane_roots:{name}"]
    return None, [f"host_resident_input_unresolved:{name}"]


def resolve_host_resident_bundle_receipt(
    receipt_path: str | Path,
    *,
    roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Resolve a bundle receipt against this host, or refuse to call it ready.

    Returns the receipt rewritten with the paths that resolved here. When any
    reference does not resolve, the returned receipt's ``status`` is
    ``not_host_resident`` -- so every downstream check that admits only
    ``ready`` refuses it without needing to know this module exists.
    """

    search_roots = tuple(configured_launch_input_roots() if roots is None else roots)
    source = Path(receipt_path).expanduser()
    blockers: list[str] = []
    if source.is_symlink() or not source.is_file():
        raise HostResidentInputError(f"host_resident_receipt_missing:{source.name}")
    resolved_source = source.resolve()
    if not path_is_host_resident(resolved_source, search_roots):
        # A receipt read out of an authoring tree is the defect this exists to
        # catch; its own location is as load-bearing as the bytes it names.
        blockers.append("host_resident_input_outside_control_plane_roots:receipt")
    try:
        receipt = json.loads(resolved_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HostResidentInputError(
            f"host_resident_receipt_unreadable:{source.name}"
        ) from exc
    if not isinstance(receipt, Mapping):
        raise HostResidentInputError(f"host_resident_receipt_not_an_object:{source.name}")

    resolved_receipt = _launch_receipt_view(receipt)
    resolutions: dict[str, Any] = {}
    for name, record in _reference_records(resolved_receipt).items():
        resolution, reference_blockers = _resolve_reference(
            name, record, receipt_dir=resolved_source.parent, roots=search_roots
        )
        blockers.extend(reference_blockers)
        if resolution is None:
            continue
        resolutions[name] = resolution
        if name == _BUNDLE_REFERENCE:
            resolved_receipt["bundle_path"] = resolution["path"]
        elif isinstance(resolved_receipt.get(name), Mapping):
            resolved_receipt[name] = {
                **dict(resolved_receipt[name]),
                "path": resolution["path"],
            }

    blockers = sorted(set(blockers))
    if blockers:
        resolved_receipt["status"] = "not_host_resident"
    return {
        "schema_version": RESOLUTION_SCHEMA,
        "status": "ready" if not blockers else "blocked",
        "receipt_path": str(resolved_source),
        "receipt_sha256": _sha256(resolved_source),
        "control_plane_roots": [str(root) for root in search_roots],
        "resolutions": resolutions,
        "receipt": resolved_receipt,
        "blockers": blockers,
        "provider_mutation_performed": False,
    }


def _profile_bound_paths(profile: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Every filesystem path a profile binds, named for the blocker it raises.

    Allocator argv is included because it is how the retained-scene lane names
    its receipt and its attempt authority, and nothing else validated it -- the
    2026-08-12 run reached the provider with two authoring paths in argv.
    Run-root placeholders are outputs the launch creates and are skipped.
    """

    bound: list[tuple[str, str]] = []
    for item in profile.get("immutable_inputs") or []:
        if not isinstance(item, Mapping):
            continue
        path = str(item.get("path") or "").strip()
        if path:
            bound.append((f"immutable_input:{item.get('name') or 'invalid'}", path))
    allocator = profile.get("allocator")
    if isinstance(allocator, Mapping):
        argv = allocator.get("argv")
        if isinstance(argv, list):
            for index, item in enumerate(argv):
                if not isinstance(item, str) or not item.startswith("/") or "{" in item:
                    continue
                flag = argv[index - 1] if index and isinstance(argv[index - 1], str) else ""
                label = flag.lstrip("-") if flag.startswith("--") else f"argv[{index}]"
                bound.append((f"allocator_argument:{label}", item))
    return bound


def launch_profile_residency_blockers(
    profile: Mapping[str, Any],
    *,
    roots: Sequence[Path] | None = None,
) -> list[str]:
    """Fail closed on a profile that binds another machine's filesystem."""

    search_roots = tuple(configured_launch_input_roots() if roots is None else roots)
    if not search_roots:
        return []
    blockers = [
        f"launch_profile_input_not_host_resident:{name}"
        for name, path in _profile_bound_paths(profile)
        if not path_is_host_resident(path, search_roots)
    ]
    return sorted(set(blockers))


def published_profile_residency_report(
    profile_dir: str | Path,
    *,
    roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Report which published profiles could still start a run on this host."""

    search_roots = tuple(configured_launch_input_roots() if roots is None else roots)
    root = Path(profile_dir).expanduser()
    if not root.is_dir():
        return {
            "schema_version": RESOLUTION_SCHEMA,
            "status": "not_configured",
            "profiles": [],
            "blockers": [],
            "provider_mutation_performed": False,
        }
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for path in sorted(root.glob("*.json")):
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            rows.append({"profile_id": path.stem, "status": "unreadable", "blockers": []})
            continue
        if not isinstance(profile, Mapping):
            rows.append({"profile_id": path.stem, "status": "unreadable", "blockers": []})
            continue
        admission = profile.get("execution_admission")
        live = bool(isinstance(admission, Mapping) and admission.get("live_enabled"))
        profile_blockers = launch_profile_residency_blockers(profile, roots=search_roots)
        rows.append(
            {
                "profile_id": str(profile.get("profile_id") or path.stem),
                "live_enabled": live,
                "status": "host_resident" if not profile_blockers else "not_host_resident",
                "blockers": profile_blockers,
            }
        )
        # A profile that cannot start a paid run is a latent defect, not a live
        # one; only a live-enabled profile blocks the guard.
        if live and profile_blockers:
            blockers.extend(
                f"{blocker}:{profile.get('profile_id') or path.stem}"
                for blocker in profile_blockers
            )
    return {
        "schema_version": RESOLUTION_SCHEMA,
        "status": "ready" if not blockers else "blocked",
        "profile_dir": str(root.resolve()),
        "control_plane_roots": [str(item) for item in search_roots],
        "profiles": rows,
        "blockers": sorted(set(blockers)),
        "provider_mutation_performed": False,
    }


__all__ = [
    "LAUNCH_INPUT_ROOTS_ENV",
    "PRODUCTION_LAUNCH_INPUT_ROOTS",
    "RESOLUTION_SCHEMA",
    "HostResidentInputError",
    "configured_launch_input_roots",
    "launch_profile_residency_blockers",
    "path_is_host_resident",
    "published_profile_residency_report",
    "resolve_host_resident_bundle_receipt",
]
