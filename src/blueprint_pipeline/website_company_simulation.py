"""The simulation package a deployment company downloads: one zip per job, made when the job's simulation exists.

Owner decision 2026-11-12: the simulation is core, because a deployment company cannot say "can do" without
testing on the twin; it downloads the simulation first and tests in its own tools (connecting its software to
Blueprint's simulation comes later and is not built here). This makes the package and publishes it exactly as the
company clip is published (``website_company_clip``):

- the source is the job's compiled team scene half, the simulation the trial itself runs: the team scene packet
  queue's ``status/<key>.json`` (``completed``, ``outcome: compiled``) names the half, whose
  ``scene/native-task-packet/`` holds ``native_task_arena_scene_plan.v1.json`` and the staged USD files under
  ``assets/``: the scene's look (``scene_appearance``), its collision (``scene_collision``) and the job's object
  (``task_object``) with its start pose, and the plan's ``task_spec.task_success_contract.criteria``;
- the package holds ``scene.usda``, written by ``scene_package_stage`` from the plan: every part referenced by a
  relative path and placed at the plan's pose under a neutral prim name (``/World/Room``, ``/World/Object``, ...),
  in metres with the plan's up axis; the parts (``scene/room.*``, ``scene/room_collision.*``, ``object.*``), what
  each needs (renamed neutrally under ``scene/files/``), ``success_rule.json`` (the criteria and run facts, neutral
  only) and ``README.txt``. No record, manifest, ledger or receipt goes in, the plan included; no file name carries
  an id. Every outside or absolute USD reference is rewritten to a path inside the package, or the package is
  refused; a plan row that cannot be placed refuses it, naming the row's role;
- every file is read with the shared outbound scan (``outbound_file_scan``: ids in both forms, account ids,
  keys, host paths, storage, file and web links, bare web addresses, and the text form of every binary USD
  layer) before the zip is made; a finding refuses the package (``refused``), naming the file and the kind;
- the zip is deterministic (names in order, fixed times and modes), so the same inputs give the same digest; it
  goes to the private artifact store by digest under a name for the capture (withdrawal deletion finds every
  version by that name), the WebApp is told its identity through the signed ``company-simulation`` control
  operation and reads it through the signed ``/api/live-pipeline/website-company-simulations`` route.

Rights are the company clip's: not withdrawn, the delivery-time takedown gate allows it, the WebApp's live task
context grants consent, and ``sharing_statement.sharing_basis`` gives a basis. The not-permitted live read is
shared with the clip (its ``statement_check.json``), so a capture that may not be shared is read once a window,
not once per package.

It runs in the Atlas preview timer's pass, after the clip (``website_atlas_preview.company_simulation_tick``):
local files and ``pxr`` only, no model, provider or paid resource. Nothing here raises out of the pass or holds a
job: every outcome is a returned status, retried on the next pass.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import math
import os
import re
import shutil
import stat
import time
import zipfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from . import company_share_permission as permission
from . import website_company_clip as clip
from .common import write_json
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .outbound_file_scan import capture_site_identity
from .website_atlas_preview_store import capture_name_matches_request, capture_withdrawn

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA = "website_company_simulation_manifest.v1"
PACKAGE_SCHEMA = "website_company_simulation.v1"
ACCEPTANCE_SCHEMA = "website_company_simulation_acceptance.v1"
DIRECTORY = ("pipeline", "company_simulation")
MANIFEST_FILE = "company_simulation.json"
PACKAGE_FILE = "simulation.zip"
STAGING = ".staging"
#: Withdrawal deletion holds this lock, so no pass publishes between its listing and deletion.
LOCK_FILE = "company_simulation.lock"
ARTIFACT_KIND = "website-company-simulation"
FORMAT = "zip_usd_scene_object_success_rule"
CONTENT_TYPE = "application/zip"
#: The one real compiled half on the host (capture-447, 2026-11-12) stages 211 MiB of USD (appearance 124 MiB,
#: collision 83 MiB, object 3.8 MiB); its configured bundle's text collision alone is 278 MiB. 512 MiB is about
#: 2.4 times the real scene, the size the first-site team runbook already allows for one flattened USD, and a
#: quarter of what the website accepts (2 GiB). Measured on the source files and on the zip.
MAX_PACKAGE_BYTES = 512 * 1024 * 1024
BUILDER_REVISION = "website_company_simulation_builder.v1"
OVER_CAP = "website_company_simulation_over_size_cap"
BUILD_TIMED_OUT = "website_company_simulation_build_timed_out"
#: One build's deadline. A real compiled half (211 MiB of USD) copies, scans by spec walk, proves and zips in
#: well under this; one that does not is recorded and tried again after the live-statement window.
BUILD_DEADLINE_SECONDS = 900
#: The pass starts no capture after this. It runs after the company clip's pass (up to 900 s, plus one encode of
#: at most 900 s) in the same unit, whose start timeout is 2 h: 600 s plus one build keeps both well inside it.
TICK_BUDGET_SECONDS = 600
QUEUE_ROOT_ENV = "BLUEPRINT_TEAM_SCENE_PACKET_QUEUE_ROOT"
DEFAULT_QUEUE_ROOT = "/var/lib/blueprint/pipeline-control-plane/team-scene-packets"
PACKET_DIRECTORIES = ("native-task-packet", "native-task-packet-three-view")
PLAN_FILE = "native_task_arena_scene_plan.v1.json"
_USD = frozenset({".usd", ".usda", ".usdc", ".usdz"})
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_RIGHTS_REFUSALS = re.compile(r"(source_revoked|consent_revoked|consent_expired|website_capture_withdrawn"
                              r"|capture_revoked_or_expired|website_company_simulation_not_permitted"
                              r"|website_company_clip_not_permitted)\b")
#: Record fields whose string values are account or owner ids, and those that are keys.
_ACCOUNT_KEYS = frozenset({"accepted_by", "account_id", "user_id", "uid", "owner_id", "owner_uid", "created_by",
                           "submitted_by", "approved_by", "requested_by", "buyer_account_id", "organization_id"})
_KEY_KEYS = frozenset({"delivery_key", "producer_delivery_key", "upload_token", "token", "session_token"})


class CompanySimulationUnavailable(ValueError):
    """No archived package this caller may read; never says why."""


class CompanySimulationBusy(RuntimeError):
    """Another pass (or withdrawal deletion) holds this capture's package lock."""


class PackageRefused(ValueError):
    """The package may not leave: a scan finding, an outside reference or an unusable scene file."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


def package_filename(capture_id: str, request_id: str) -> str:
    """The archive name: the walkthrough's own, or a continuation scoped to its job."""
    if not capture_name_matches_request(request_id, capture_id):
        raise CompanySimulationUnavailable("website_company_simulation_unavailable")
    return f"{capture_id}.zip" if capture_id.startswith("walkthrough-") else f"{request_id}-{capture_id}.zip"


def _read(path: Path) -> dict[str, Any]:
    return permission.read_json(Path(path))


def _sha256_file(path: Path) -> str:
    return permission.sha256_file(Path(path))


# --- where the job's simulation is -------------------------------------------------------------------------------

def _queue_root() -> Path:
    return Path(os.environ.get(QUEUE_ROOT_ENV) or DEFAULT_QUEUE_ROOT)


def _inside(base: Path, path: Path) -> Path | None:
    """``path`` resolved, when it is a regular file under ``base`` reached through no link."""
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    if any(item.is_symlink() for item in (path, *path.parents) if item.is_relative_to(base)) \
            or not resolved.is_relative_to(base.resolve()) or not resolved.is_file():
        return None
    return resolved


#: The team scene packet worker's two outcomes of a completed status that name a half
#: (``team_scene_packet_worker``: ``_compiled_status`` writes ``compiled``, a reused half ``already_compiled``).
SUCCESS_OUTCOMES = frozenset({"compiled", "already_compiled"})


def simulation_source(capture_root: Path, *, queue_root: Path | None = None) -> dict[str, Any] | None:
    """The newest compiled team scene half of this capture, as its plan and staged files name it, or None.

    None while no half is compiled (the job is never held for it). A half whose plan or files do not read is
    None too, and is tried again on the next pass.
    """
    root, queue = Path(capture_root), Path(queue_root or _queue_root())
    identity = permission.capture_identity(root)
    statuses = []
    try:
        candidates = sorted((queue / "status").glob("*.json"))
    except OSError:
        return None
    for path in candidates:
        status = _read(path)
        # A reused half (a duplicate request, or a retry that found its half) is written as ``already_compiled``;
        # either way the half itself is checked below (``_half``), never trusted for the word.
        if (status.get("status") == "completed" and status.get("outcome") in SUCCESS_OUTCOMES
                and status.get("capture_id") == identity["capture_id"]
                and Path(str(status.get("capture_root") or "")) == root
                and isinstance(status.get("team_scene_inputs_path"), str)):
            statuses.append((path.stat().st_mtime_ns, path.name, status))
    for _mtime, _name, status in sorted(statuses, reverse=True):
        source = _half(Path(status["team_scene_inputs_path"]).parent, status, estimates=object_estimates(root))
        if source is not None:
            return source
    return None


def object_estimates(capture_root: Path) -> Mapping[str, Any] | None:
    """The object's mass and friction ranges as preparation estimated them from the video, when recorded.

    Read from the preparation the scene handoff names (inside the capture), else the default preparation record.
    """
    root = Path(capture_root)
    preparation = root / "pipeline" / "website_scene_preparation" / "preparation.json"
    named = _read(root / "pipeline" / "website_scene_preparation" / "handoff.json").get("preparation_path")
    if isinstance(named, str):
        candidate = Path(named)
        candidate = candidate if candidate.is_absolute() else root / candidate
        if _inside(root, candidate) is not None:
            preparation = candidate
    bounds = (_read(preparation).get("physics") or {}).get("bounds")
    return bounds if isinstance(bounds, Mapping) else None


def _half(scene: Path, status: Mapping[str, Any], *, estimates: Mapping[str, Any] | None = None
          ) -> dict[str, Any] | None:
    """The half's plan and shipped files; a row that cannot be placed is kept as ``refusal`` naming its role."""
    from .scene_package_stage import StageRefused, neutral_rule, placements, up_axis
    if not scene.is_absolute() or scene.is_symlink() or not scene.is_dir():
        return None
    for name in PACKET_DIRECTORIES:
        packet = scene / name
        plan_path = _inside(scene, packet / PLAN_FILE)
        if plan_path is None:
            continue
        plan = _read(plan_path)
        if not plan:
            continue
        source: dict[str, Any] = {"packet_dir": packet, "plan_path": plan_path, "plan": plan,
                                  "plan_digest": plan.get("plan_digest"), "status": dict(status), "files": {},
                                  "refusal": None}
        try:
            rows = placements(plan)
            axis = up_axis(plan)
            rule = neutral_rule(plan, estimates=estimates)
            if rule is None:
                raise StageRefused("scene_stage_success_rule_missing")
            for placement, raw in zip(rows, plan["objects"], strict=True):
                role = placement["role"]
                path = _inside(packet, packet.joinpath(*PurePosixPath(placement["usd_path"]).parts))
                if (path is None or path.stat().st_size != raw.get("size_bytes")
                        or not _DIGEST.fullmatch(str(raw.get("sha256") or ""))):
                    raise StageRefused("scene_stage_asset_not_shipped", role)
                source["files"][placement["prim"]] = {"path": path, "role": role, "sha256": raw["sha256"], "size_bytes": int(raw["size_bytes"]),
                                         "suffix": PurePosixPath(placement["usd_path"]).suffix.lower(),
                                         "placement": {key: value for key, value in placement.items()
                                                       if key != "usd_path"}}
            source.update(axis=axis, rule=rule, assets_dir=packet / str(plan.get("asset_directory") or "assets"),
                          appearance_representation=(plan.get("appearance_frame_alignment") or {}).get(
                              "representation"))
        except StageRefused as error:
            source["refusal"] = "website_company_simulation_" + str(error).removeprefix("scene_stage_")
        return source
    return None


def simulation_revision(source: Mapping[str, Any], *, max_bytes: int = MAX_PACKAGE_BYTES) -> str:
    """Names this package: the files' digests, their placements, the success rule, the builder and the cap."""
    return canonical_digest({
        "builder": BUILDER_REVISION, "format": FORMAT, "max_bytes": int(max_bytes),
        "files": {prim: {key: row[key] for key in ("sha256", "size_bytes", "suffix", "placement")}
                  for prim, row in sorted(source["files"].items())},
        "rule": source.get("rule"), "axis": source.get("axis"), "refusal": source.get("refusal"),
        "plan_digest": source.get("plan_digest") if source.get("refusal") else None})


# --- what must never leave --------------------------------------------------------------------------------------

def _walk_values(value: Any, keys: frozenset[str]) -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in keys and isinstance(item, str):
                yield item
            else:
                yield from _walk_values(item, keys)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item, keys)


def secret_literals(capture_root: Path, source: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The job's own ids, account ids and keys, each with the kind printed in a finding (never the value)."""
    root = Path(capture_root)
    identity = permission.capture_identity(root)
    status = source.get("status") or {}
    records = [_read(root / name) for name in (
        "capture_descriptor.json", "raw/manifest.json", "raw/capture_upload_complete.json",
        "pipeline_staging_manifest.json", "pipeline_job_ledger.json", "pipeline/website_task_context.json")]
    records.append(_read(Path(source["plan_path"])))
    found = [(identity[key], "a job id") for key in ("request_id", "scene_id", "capture_id")]
    core = re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", identity["request_id"])
    if core:
        found += [(core.group(0), "a job id"), (core.group(0).replace("-", ""), "a job id")]
    found += [(str(status.get(key) or ""), "a job id") for key in ("task_id", "request_key")]
    for record in records:
        found += [(value, "a job id") for value in _walk_values(record, frozenset({"site_submission_id", "intent_id",
                                                                                     "scene_intent_id", "site_id"}))]
        found += [(value, "an account id") for value in _walk_values(record, _ACCOUNT_KEYS)]
        found += [(value, "a delivery key") for value in _walk_values(record, _KEY_KEYS)]
    # Short words ("system", "operator") are roles, not ids: an id here is at least eight characters.
    return [(value, kind) for value, kind in found if len(value) >= 8 and not re.fullmatch(r"[a-z_]+", value)]


# --- making the package ----------------------------------------------------------------------------------------

def _open_layer(path: str) -> Any:
    """The layer, or None when ``pxr`` cannot open it (it raises for a broken file)."""
    from pxr import Sdf
    try:
        return Sdf.Layer.OpenAsAnonymous(path)
    except Exception:  # noqa: BLE001 - any failure to open is a refusal by the caller
        return None


def _asset_paths(layer: Any) -> list[str]:
    from pxr import UsdUtils
    found: list[str] = []

    def collect(path: str) -> str:
        found.append(path)
        return path
    UsdUtils.ModifyAssetPaths(layer, collect)
    return found


def _outside(path: str) -> bool:
    pure = PurePosixPath(path.replace("\\", "/"))
    return ("://" in path or path.startswith(("file:", "/", "\\")) or re.match(r"[A-Za-z]:", path) is not None
            or ".." in pure.parts)


def _check_usdz(path: Path, label: str) -> None:
    """A usdz is kept as it is: every reference inside it must stay inside it."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = [info.filename for info in archive.infolist() if Path(info.filename).suffix.lower() in _USD]
    except (OSError, zipfile.BadZipFile):
        raise PackageRefused("website_company_simulation_scene_unreadable", label) from None
    if not names:
        raise PackageRefused("website_company_simulation_scene_unreadable", label)
    for name in names:
        layer = _open_layer(f"{path}[{name}]")
        if layer is None:
            raise PackageRefused("website_company_simulation_scene_unreadable", label)
        for reference in [*_asset_paths(layer), *(value for _spec, value in _file_strings(layer))]:
            if reference and _outside(reference):
                raise PackageRefused("website_company_simulation_outside_reference", label)


#: A string or token value with one of these endings names a file: it is shipped and rewritten like an asset path.
FILE_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz", ".png", ".jpg", ".jpeg", ".exr", ".hdr", ".tga",
                           ".tif", ".tiff", ".bmp", ".ktx", ".ktx2", ".dds", ".mdl", ".obj", ".ply", ".glb", ".gltf",
                           ".vdb", ".nvdb", ".mtl", ".wav", ".mp4", ".mov"})
_STRING_TYPES = frozenset({"string", "token", "string[]", "token[]"})


def _names_file(value: Any) -> bool:
    return isinstance(value, str) and PurePosixPath(value.strip().replace("\\", "/")).suffix.lower() in FILE_SUFFIXES


def _file_strings(layer: Any) -> list[tuple[Any, str]]:
    """Every (attribute spec, value) whose string or token value names a file, in defaults and time samples."""
    from pxr import Sdf
    found: list[tuple[Any, str]] = []
    paths: list = []
    layer.Traverse(Sdf.Path.absoluteRootPath, paths.append)
    for path in paths:
        spec = layer.GetAttributeAtPath(path) if path.IsPropertyPath() else None
        if spec is None or str(spec.typeName) not in _STRING_TYPES:
            continue
        values = [spec.default] if spec.HasDefaultValue() else []
        values += [layer.QueryTimeSample(path, sample) for sample in layer.ListTimeSamplesForPath(path)]
        for value in values:
            for item in (list(value) if not isinstance(value, str) and value is not None else [value]):
                if _names_file(item):
                    found.append((spec, str(item)))
    return found


def _rewrite_strings(layer: Any, rewrite: Callable[[str], str]) -> None:
    """Rewrite every string or token value that names a file, in defaults and time samples."""
    from pxr import Sdf
    paths: list = []
    layer.Traverse(Sdf.Path.absoluteRootPath, paths.append)

    def fixed(value: Any) -> Any:
        if isinstance(value, str):
            return rewrite(value) if _names_file(value) else value
        if value is not None and not isinstance(value, str):
            items = list(value)
            new = [rewrite(item) if _names_file(item) else item for item in items]
            return new if new != items else value
        return value
    for path in paths:
        spec = layer.GetAttributeAtPath(path) if path.IsPropertyPath() else None
        if spec is None or str(spec.typeName) not in _STRING_TYPES:
            continue
        if spec.HasDefaultValue():
            new = fixed(spec.default)
            if new is not spec.default and new != spec.default:
                spec.default = new
        for sample in layer.ListTimeSamplesForPath(path):
            value = layer.QueryTimeSample(path, sample)
            new = fixed(value)
            if new is not value and new != value:
                layer.SetTimeSample(path, sample, new)


_UDIM = "<UDIM>"
#: The closed list of kinds a scene may ship (``outbound_file_scan``, Rule 1), by ending and by content.
_KIND_BY_SUFFIX = {".usd": ("usdc", "usda"), ".usda": ("usda",), ".usdc": ("usdc",), ".usdz": ("usdz",),
                   ".png": ("png",), ".jpg": ("jpeg",), ".jpeg": ("jpeg",)}
KIND_NOT_SHIPPED = "website_company_simulation_file_kind_not_shipped"


def _shippable(path: Path, label: str) -> None:
    """Refuse, naming the role and the kind, a file a scene depends on that is not on the closed list."""
    from .outbound_file_scan import content_kind
    suffix = path.suffix.lower()
    allowed = _KIND_BY_SUFFIX.get(suffix)
    try:
        actual = content_kind(path)
    except OSError:
        actual = None
    if allowed is None or actual not in allowed:
        raise PackageRefused(KIND_NOT_SHIPPED, f"{label} {suffix or 'no ending'}")


class _Localizer:
    """Copies each USD file into the package, rewriting every reference to a relative path inside it.

    Asset paths (sublayers, references, payloads, asset-valued attributes) and string or token values that name a
    file are rewritten; sublayer time offsets are kept; every tile of a UDIM texture inside the scene's own folder
    is shipped. A reference that is a link, points outside the scene's files or names nothing refuses the package.
    """

    def __init__(self, assets_dir: Path, staging: Path):
        self.assets, self.staging = assets_dir.resolve(), staging
        self.placed: dict[Path, Path] = {}
        self.count = 0

    def _target(self, suffix: str) -> Path:
        self.count += 1
        return self.staging / "scene" / "files" / f"file-{self.count:03d}{suffix}"

    def place(self, source: Path, target: Path, label: str) -> None:
        source = source.resolve()
        _shippable(source, label)
        self.placed[source] = target
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() == ".usdz":
            _check_usdz(source, label)
            shutil.copyfile(source, target)
            return
        if source.suffix.lower() not in _USD:
            shutil.copyfile(source, target)
            return
        layer = _open_layer(str(source))
        if layer is None:
            raise PackageRefused("website_company_simulation_scene_unreadable", label)
        changed: list[bool] = []
        failure: list[str] = []

        def inside(candidate: Path) -> Path | None:
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                return None
            return resolved if resolved.is_relative_to(self.assets) and resolved.is_file() else None

        def rewrite(reference: str) -> str:
            if not reference or failure:
                return reference
            if "://" in reference or reference.startswith("file:"):
                failure.append("website_company_simulation_outside_reference")
                return reference
            base = Path(reference) if Path(reference).is_absolute() else source.parent / reference
            if _UDIM in reference:
                return udim(reference, base)
            try:
                base.resolve(strict=True)
            except OSError:
                failure.append("website_company_simulation_reference_missing")
                return reference
            resolved = inside(base)
            if resolved is None:
                failure.append("website_company_simulation_outside_reference")
                return reference
            placed = self.placed.get(resolved)
            if placed is None:
                placed = self._target(resolved.suffix.lower())
                self.place(resolved, placed, label)
            relative = "./" + Path(os.path.relpath(placed, target.parent)).as_posix()
            if relative != reference:
                changed.append(True)
            return relative

        def udim(reference: str, base: Path) -> str:
            head, _, tail = base.name.partition(_UDIM)
            pattern = re.compile(re.escape(head) + r"(1[0-9]{3})" + re.escape(tail))
            try:
                tiles = sorted(item for item in base.parent.iterdir() if pattern.fullmatch(item.name))
            except OSError:
                tiles = []
            if not tiles or any(inside(tile) is None for tile in tiles):
                failure.append("website_company_simulation_reference_missing" if not tiles
                               else "website_company_simulation_outside_reference")
                return reference
            stem = self._target("")
            for tile in tiles:
                _shippable(tile.resolve(), label)
                number = pattern.fullmatch(tile.name).group(1)
                tile_target = stem.with_name(f"{stem.name}.{number}{tail.lower() or tile.suffix.lower()}")
                tile_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(tile.resolve(), tile_target)
                self.placed.setdefault(tile.resolve(), tile_target)
            changed.append(True)
            return "./" + Path(os.path.relpath(stem.with_name(f"{stem.name}.{_UDIM}{tail.lower()}"),
                                                target.parent)).as_posix()
        from pxr import UsdUtils
        offsets = list(layer.subLayerOffsets)
        UsdUtils.ModifyAssetPaths(layer, rewrite)
        for index, offset in enumerate(offsets):
            if index < len(layer.subLayerPaths):
                layer.subLayerOffsets[index] = offset
        if not failure:
            _rewrite_strings(layer, rewrite)
        if failure:
            raise PackageRefused(failure[0], label)
        if changed:
            if not layer.Export(str(target)):
                raise PackageRefused("website_company_simulation_scene_unreadable", label)
        else:
            shutil.copyfile(source, target)


def _layer_units(path: Path) -> tuple[str | None, float | None]:
    layer = _open_layer(str(path))
    root = layer.pseudoRoot if layer is not None else None
    up = root.GetInfo("upAxis") if root is not None and root.HasInfo("upAxis") else None
    metres = root.GetInfo("metersPerUnit") if root is not None and root.HasInfo("metersPerUnit") else None
    return (str(up) if up else None), (float(metres) if metres is not None else None)


_SAYS = {"scene_appearance": "how the site looks", "scene_collision": "the site's surfaces for contact",
         "task_object": "the job's object, at its start position", "task_support": "the surface the object goes on",
         "replacement": "a stand-in part of the site"}


def _root_offset(path: Path) -> dict[str, list[float]] | None:
    """The part file's own root translation and rotation, when either is not identity (1 mm, 0.001 rad)."""
    from pxr import Gf, Usd, UsdGeom
    try:
        stage = Usd.Stage.Open(str(path), Usd.Stage.LoadNone)
        prim = stage.GetDefaultPrim() or next(iter(stage.GetPseudoRoot().GetChildren()), None)
        if prim is None or not prim.IsA(UsdGeom.Xformable):
            return None
        matrix = UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default())
        transform = Gf.Transform(matrix)
    except Exception:  # noqa: BLE001 - an unreadable part is refused elsewhere; no note
        return None
    translation = [float(value) for value in transform.GetTranslation()]
    quat = transform.GetRotation().GetQuat()
    xyzw = [float(value) for value in quat.GetImaginary()] + [float(quat.GetReal())]
    angle = transform.GetRotation().GetAngle()
    if max(abs(value) for value in translation) <= 1e-3 and abs(angle) <= math.degrees(1e-3):
        return None
    return {"translation_m": [round(value, 6) for value in translation],
            "rotation_xyzw": [round(value, 6) for value in xyzw]}


def _readme(source: Mapping[str, Any], placed: Mapping[str, str], names: list[str],
            units: Mapping[str, tuple[str | None, float | None]],
            offsets: Mapping[str, Any] | None = None) -> str:
    """Only what is true of the package in front of it."""
    lines = ["Simulation of one job", "", "What this is", "",
             "A simulation of the site a deployment company was asked about, with the job's object in it,",
             "and the rule for what counts as the job done. Test in your own tools.", "",
             "How the pieces fit", "",
             "- scene.usda: open this one. It places every part below at its position, in /World."]
    hidden = []
    for prim, relative in placed.items():
        row = source["files"][prim]
        says = _SAYS.get(row["role"], "a part of the site")
        visible = row["placement"]["visible"]
        lines.append(f"- {relative} (/World/{prim}): {says}." + ("" if visible else " Hidden in scene.usda."))
        if not visible:
            hidden.append(prim)
    if any(name.startswith("scene/files/") for name in names):
        lines.append("- scene/files/: files the parts above use.")
    lines += ["- success_rule.json: what counts as done, as numbers, with the run's timing and the object's",
              "  start position and size where known.", "- README.txt: this file.", ""]
    lines += ["Each part's file is referenced on a child prim, /World/<part>/Asset, under the part's position and",
              "orientation in the simulation, which are on /World/<part>. The file keeps its own root transform."]
    for prim, note in sorted((offsets or {}).items()):
        lines += [f"/World/{prim}: this part's file carries its own offset (its root prim is moved or turned",
                  "inside the file; see part_notes in success_rule.json). Where a tool places a file's root at the",
                  "part's position instead of under it, this part lands somewhere else: check it."]
    for prim in hidden:
        lines += [f"/World/{prim} is in the scene but hidden, as in Blueprint's simulation, where it is used for",
                  "contact and not drawn. To see it, set its visibility to inherited (in usdview: select it and",
                  "press Make Visible)."]
    lines += ["", "Units and axes", "", "- scene.usda is in metres (1 unit is 1 metre), radians and seconds.",
              f"- Up is +{source['axis']} in scene.usda."]
    for prim, relative in placed.items():
        up, metres = units[prim]
        stated = [f"up axis {up}" if up else None, f"{metres:g} metres per unit" if metres is not None else None]
        stated = [item for item in stated if item]
        lines.append(f"- {relative} states " + (" and ".join(stated) if stated else "no up axis or unit") + ".")
    if source.get("appearance_representation") == "particlefield_3d_gaussian_splat":
        lines += ["- The room's look is a 3D Gaussian splat stored as USD particle fields. Some tools cannot show",
                  "  it; the hidden collision part holds the room's geometry."]
    lines += ["", "How it was made", "",
              "Built from one phone video of the site. Sizes, distances and positions are estimates from that",
              "video, not measurements. A result in this simulation is not a promise of what a robot does at the",
              "site, and says nothing about safety. No robot is included: bring your own."]
    if (source.get("rule") or {}).get("object_estimates_from_video"):
        lines.append("The object's mass and friction in success_rule.json are ranges estimated from the video.")
    lines += ["", "Terms", "",
              "This package is for judging this one job. Do not pass it on and do not use it to train software.",
              "Blueprint may ask you to delete it; then delete every copy.", ""]
    return "\n".join(lines)


def build_package(source: Mapping[str, Any], output: Path, *, literals: list[tuple[str, str]],
                  words: list[tuple[bytes, str]] | None = None, max_bytes: int = MAX_PACKAGE_BYTES,
                  deadline: float | None = None) -> dict[str, Any]:
    """Stage, rewrite, compose, scan and zip the package to ``output``; ``PackageRefused`` or ``OVER_CAP`` otherwise.

    It holds no lock: every file it writes is its own (a staging folder and a partial zip named for this build).
    ``deadline`` (a ``time.monotonic`` value) is checked between steps and files: past it, ``BUILD_TIMED_OUT``.
    """
    def on_time() -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise ValueError(BUILD_TIMED_OUT)
    import tempfile

    from .outbound_file_scan import literal_forms, scan_tree
    from .scene_package_stage import StageRefused, write_stage
    if source.get("refusal"):
        raise PackageRefused(source["refusal"])
    if sum(row["size_bytes"] for row in source["files"].values()) > max_bytes:
        raise ValueError(OVER_CAP)
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=STAGING + "-", dir=output.parent))
    staging, scratch = work / "package", work / "scan"
    partial = work / "simulation.zip.partial"
    staging.mkdir(mode=0o750)
    scratch.mkdir(mode=0o700)
    try:
        localizer = _Localizer(Path(source["assets_dir"]), staging)
        placed: dict[str, str] = {}
        units: dict[str, tuple[str | None, float | None]] = {}
        offsets: dict[str, dict[str, list[float]]] = {}
        rows = []
        for prim, row in source["files"].items():
            on_time()
            if _sha256_file(row["path"]) != row["sha256"]:
                raise PackageRefused("website_company_simulation_source_changed", row["role"])
            relative = row["placement"]["name"] + row["suffix"]
            localizer.place(Path(row["path"]), staging / relative, row["role"])
            placed[prim] = relative
            units[prim] = _layer_units(staging / relative)
            offset = _root_offset(staging / relative)
            if offset is not None:
                offsets[prim] = offset
            rows.append({**row["placement"], "file": relative})
        try:
            write_stage(staging / "scene.usda", rows, axis=source["axis"])
        except StageRefused as error:
            raise PackageRefused("website_company_simulation_" + str(error).removeprefix("scene_stage_")) from None
        rule = dict(source["rule"])
        if offsets:
            rule["part_notes"] = {f"/World/{prim}": {"own_root_offset": offset}
                                  for prim, offset in sorted(offsets.items())}
        (staging / "success_rule.json").write_text(json.dumps(rule, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
        names = sorted(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file())
        (staging / "README.txt").write_text(_readme(source, placed, [*names, "README.txt"], units, offsets), encoding="utf-8")
        on_time()
        _resolves_inside(staging)
        findings = scan_tree(staging, [*literal_forms(literals), *(words or [])], scratch=scratch, before_each=on_time,
                             words_except=("README.txt", "scene.usda"), closed=True)
        on_time()
        from .outbound_file_scan import NOT_SHIPPED
        if findings and findings[0].kind == NOT_SHIPPED:
            raise PackageRefused(KIND_NOT_SHIPPED, findings[0].file)
        if findings:
            raise PackageRefused("website_company_simulation_file_names_the_site", findings[0].sentence())
        names = sorted(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file())
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in names:
                info = zipfile.ZipInfo(name, date_time=_FIXED_TIME)
                info.compress_type, info.create_system, info.external_attr = zipfile.ZIP_DEFLATED, 3, 0o100644 << 16
                with (staging / name).open("rb") as stream, archive.open(info, "w", force_zip64=True) as member:
                    shutil.copyfileobj(stream, member, 8 << 20)
        size = partial.stat().st_size
        if size > max_bytes:
            raise ValueError(OVER_CAP)
        os.replace(partial, output)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"format": FORMAT, "content_type": CONTENT_TYPE, "digest": _sha256_file(output), "size_bytes": size,
            "files": names, "max_bytes": int(max_bytes)}


def _resolves_inside(staging: Path) -> None:
    """``scene.usda`` opens, and every layer and file it uses resolves inside the package."""
    from pxr import Usd, UsdUtils
    scene = staging / "scene.usda"
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(scene))
        stage = Usd.Stage.Open(str(scene), Usd.Stage.LoadNone)
    except Exception:  # noqa: BLE001 - any failure is a refusal
        raise PackageRefused("website_company_simulation_scene_unreadable", "scene.usda") from None
    if stage is None or unresolved or not stage.GetPrimAtPath("/World/Object").IsValid():
        raise PackageRefused("website_company_simulation_reference_missing", "scene.usda")
    for item in [*(layer.realPath for layer in layers), *assets]:
        path = Path(str(item).split("[", 1)[0])
        if not path.resolve().is_relative_to(staging.resolve()):
            raise PackageRefused("website_company_simulation_outside_reference", "scene.usda")


# --- publishing -------------------------------------------------------------------------------------------------

def package_record(*, identity: Mapping[str, str], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """What the WebApp may store: the package's identity, never a path, object key or person."""
    built, basis = manifest["package"], manifest["permission"]
    value = {"schema_version": PACKAGE_SCHEMA, "capture_id": identity["capture_id"],
             "simulation_revision": manifest["simulation_revision"], "state": "ready", "format": built["format"],
             "content_type": built["content_type"], "digest": built["digest"], "size_bytes": built["size_bytes"],
             "files": list(built["files"]), "permission_basis": basis["permission_basis"],
             "statement_version": basis["statement_version"], "updated_at_epoch": int(manifest["built_at_epoch"])}
    value["simulation_digest"] = cross_runtime_canonical_digest(value)
    return value


@contextmanager
def package_lock(capture_root: Path) -> Iterator[None]:
    directory = Path(capture_root).joinpath(*DIRECTORY)
    if not directory.parent.is_dir() or directory.parent.is_symlink():
        raise ValueError("website_company_simulation_capture_changed")  # a retired capture is never re-created
    directory.mkdir(mode=0o750, exist_ok=True)
    descriptor = os.open(directory / LOCK_FILE, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError("website_company_simulation_lock_unsafe")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CompanySimulationBusy("website_company_simulation_busy") from None
        yield


def _archive(path: Path, *, identity: Mapping[str, str]) -> dict[str, Any]:
    """The private artifact store only, never the legacy fallback, with its bytes proven."""
    from .task_evaluation_configured_scene_object_store import (
        publish_configured_scene_artifact, verify_dedicated_artifact_store)
    verify_dedicated_artifact_store()
    filename = package_filename(identity["capture_id"], identity["request_id"])
    reference = publish_configured_scene_artifact(path=path, artifact_kind=ARTIFACT_KIND, filename=filename)
    return {"uri": reference["uri"], "digest": reference["digest"], "size_bytes": reference["size_bytes"],
            "filename": filename}


def _deliver(record: Mapping[str, Any], *, identity: Mapping[str, str]) -> None:
    from .website_task_context import website_webapp_request
    acceptance = website_webapp_request(capture_id=identity["capture_id"], operation="company-simulation", payload={
        "request_id": identity["request_id"], "scene_id": identity["scene_id"], "simulation": dict(record)})
    if (acceptance.get("schema_version") != ACCEPTANCE_SCHEMA or acceptance.get("accepted") is not True
            or acceptance.get("simulation_revision") != record["simulation_revision"]
            or acceptance.get("simulation_digest") != record["simulation_digest"]):
        raise ValueError("website_company_simulation_acceptance_invalid")


def _current(manifest: Mapping[str, Any], revision: str, package: Path) -> bool:
    built, archive = manifest.get("package") or {}, manifest.get("archive") or {}
    return (manifest.get("schema_version") == MANIFEST_SCHEMA and manifest.get("simulation_revision") == revision
            and not package.is_symlink() and package.is_file() and package.stat().st_size == built.get("size_bytes")
            and archive.get("digest") == built.get("digest") and not permission.hidden(archive)
            and (manifest.get("delivered") or {}).get("simulation_digest") == manifest.get("simulation_digest"))


def _take_back_if_archived(root: Path, manifest_path: Path, retract: Callable[[str], None]) -> None:
    if not manifest_path.is_file():
        return
    with package_lock(root):
        manifest = _read(manifest_path)
        if manifest:
            permission.take_back(manifest, manifest_path, retract)


#: Read at call time, so a test can point it at a temporary list (the clip's own setting when unset).
OWNED_CAPTURES_LIST: Path | None = None


def _owned_list() -> Path | None:
    return OWNED_CAPTURES_LIST if OWNED_CAPTURES_LIST is not None else clip.OWNED_CAPTURES_LIST


def upload_sha256(capture_root: Path) -> str | None:
    """The digest of the upload the scene was built from, which Blueprint's own list names by prefix: the
    original normalization recorded, else the one the company clip recorded for the raw upload."""
    recorded = clip.recorded_video_sha256(capture_root)
    if recorded:
        return recorded
    value = _read(Path(capture_root).joinpath(*clip.DIRECTORY) / clip.MANIFEST_FILE).get("upload_sha256")
    return value if isinstance(value, str) and _DIGEST.fullmatch(value) else None


def publish_company_simulation(capture_root: Path, *, now: Callable[[], float] = time.time,
                               check_rights: Callable[[Mapping[str, str]], Mapping[str, Any]] = permission.live_rights,
                               archive: Callable[..., dict[str, Any]] = _archive,
                               retract: Callable[[str], None] = permission.retract_object,
                               deliver: Callable[..., None] = _deliver, queue_root: Path | None = None,
                               max_bytes: int = MAX_PACKAGE_BYTES,
                               build_seconds: float = BUILD_DEADLINE_SECONDS) -> dict[str, Any]:
    """One idempotent pass for one capture: make, archive and deliver what is not yet current, and take back
    what may no longer be shared (``company_share_permission``, shared with the company clip).

    Returns ``{"status": ...}``: ``current``, ``published``, ``not_permitted`` (no basis; any archived copy is
    taken back), ``no_source`` (no compiled simulation yet), ``over_size_cap``, ``build_timed_out`` (tried again
    after the window), ``withdrawn``, ``refused`` (rights, the takedown gate, the website, or the package may not
    leave, with a reason), ``busy`` or ``failed`` (retried by the next pass). It never raises.
    """
    try:
        return _publish(Path(capture_root), now, check_rights, archive, retract, deliver, queue_root, max_bytes,
                        build_seconds)
    except CompanySimulationBusy:
        return {"status": "busy"}
    except Exception as error:  # noqa: BLE001 - one capture never stops a pass; retried next time
        logger.warning("website_company_simulation_failed capture=%s type=%s", Path(capture_root).name,
                       type(error).__name__)
        return {"status": "failed", "reason": _own_reason(error)}


def _own_reason(error: BaseException | str) -> str:
    """The reason in the simulation's own name: the shared rights checks speak in the clip's."""
    reason = error if isinstance(error, str) else permission.reason_code(error)
    return "website_company_simulation_" + reason.removeprefix("website_company_clip_") \
        if reason.startswith("website_company_clip_") else reason


def _own(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is not None and isinstance(result.get("reason"), str):
        result = {**result, "reason": _own_reason(result["reason"])}
    return result


def _refused_live(check, manifest, manifest_path, retract):
    return _own(permission.refused_live(check, manifest, manifest_path, retract,
                                        unanswered="website_company_simulation_rights_unanswered"))


#: Build outcomes recorded for a revision and tried again only after the window, not every pass.
_RETRIED_AFTER_WINDOW = re.compile(r"website_company_simulation_[a-z_]+")


def _publish(root, now, check_rights, archive, retract, deliver, queue_root, max_bytes, build_seconds):
    identity = permission.capture_identity(root)
    directory = root.joinpath(*DIRECTORY)
    manifest_path, package_path = directory / MANIFEST_FILE, directory / PACKAGE_FILE
    owned_list = _owned_list()
    if permission.retraction_pending(_read(manifest_path)):
        with package_lock(root):
            manifest = _read(manifest_path)
            if permission.retraction_pending(manifest) and not permission.take_back(manifest, manifest_path, retract):
                return {"status": "failed", "reason": "website_company_simulation_retraction_pending"}
    if capture_withdrawn(root):
        return {"status": "withdrawn"}
    if not permission.delivery_gate_allows(root, surface="website_company_simulation"):
        _take_back_if_archived(root, manifest_path, retract)
        return {"status": "refused", "reason": "website_company_simulation_takedown_gate_closed"}
    if not permission.may_be_permitted(root, identity, now(), owned_list):
        _take_back_if_archived(root, manifest_path, retract)
        return {"status": "not_permitted"}
    if simulation_source(root, queue_root=queue_root) is None:
        return {"status": "no_source"}
    # 1. Under the lock: decide. 2. Without it: build (minutes for a real scene; withdrawal deletion is never held).
    # 3. Under the lock again: check nothing changed, then publish.
    with package_lock(root):
        source = simulation_source(root, queue_root=queue_root)
        if source is None:
            return {"status": "no_source"}
        revision = simulation_revision(source, max_bytes=max_bytes)
        manifest = _read(manifest_path)
        upload = upload_sha256(root)
        owned = permission.owned_basis_holds(identity["capture_id"], upload, owned_list)
        check = permission.read_check(root)
        if _current(manifest, revision, package_path):
            # A published package keeps its basis or goes: Blueprint's list every pass, the live rights at least
            # once a window (and at once after a recheck mark). A definite "no" takes it back on this pass.
            if permission.live_due(check, now(), root):
                check = permission.record_live(root, identity, check_rights, now())
                if check["outcome"] == "rights_refused":
                    return _refused_live(check, manifest, manifest_path, retract)
            if check.get("outcome") == "rights_refused":  # recorded by this pass or the clip's: any basis stops
                return _refused_live(check, manifest, manifest_path, retract)
            basis = (manifest.get("permission") or {}).get("permission_basis")
            if ((basis == "blueprint_owned_capture" and owned)
                    or (basis == "site_sharing_statement" and permission.live_grant_holds(check, now()))):
                return {"status": "current", "simulation_revision": revision}
            permission.take_back(manifest, manifest_path, retract)
            return {"status": "not_permitted"}
        refused, refused_at = manifest.get("refused"), manifest.get("refused_at_epoch")
        if manifest.get("simulation_revision") == revision and refused == OVER_CAP:
            return {"status": "over_size_cap"}
        if (manifest.get("simulation_revision") == revision and isinstance(refused, str)
                and isinstance(refused_at, int) and _RETRIED_AFTER_WINDOW.fullmatch(refused)
                and 0 <= now() - refused_at < permission.LIVE_STATEMENT_RECHECK_SECONDS):
            return ({"status": "build_timed_out"} if refused == BUILD_TIMED_OUT
                    else {"status": "refused", "reason": refused})
        # A definite "no" from this window's live read stands until the window passes: no request every pass.
        if (not owned and not permission.live_due(check, now(), root)
                and check.get("outcome") in ("not_sharing", "rights_refused")):
            if manifest:
                permission.take_back(manifest, manifest_path, retract)
            return ({"status": "not_permitted"} if check["outcome"] == "not_sharing"
                    else {"status": "refused", "reason": _own_reason(check.get("reason") or permission.RIGHTS_REFUSED)})
        # A build or a delivery: always on a fresh live answer, recorded whatever it says.
        check = permission.record_live(root, identity, check_rights, now())
        stopped = _refused_live(check, manifest, manifest_path, retract)
        if stopped is not None:
            return stopped
        basis = permission.decided_basis(check, owned)
        if basis is None:
            if manifest:
                permission.take_back(manifest, manifest_path, retract)
            return {"status": "not_permitted"}
        # The website refused this package: nothing is uploaded again until the live rights differ from those at
        # the refusal, the window passes or a new revision arrives. A recheck only makes the live read due.
        memory = manifest.get("refused_by_webapp")
        if (isinstance(memory, Mapping) and memory.get("simulation_revision") == revision
                and memory.get("rights_fingerprint") == check.get("rights_fingerprint")
                and isinstance(memory.get("at_epoch"), int)
                and 0 <= now() - memory["at_epoch"] < permission.LIVE_STATEMENT_RECHECK_SECONDS):
            return {"status": "refused", "reason": str(memory.get("reason"))}
        built = manifest.get("package") or {}
        needs_build = (manifest.get("simulation_revision") != revision or package_path.is_symlink()
                       or not package_path.is_file() or _sha256_file(package_path) != built.get("digest"))
        if needs_build and manifest.get("archive") and not permission.take_back(manifest, manifest_path, retract):
            return {"status": "failed", "reason": "website_company_simulation_retraction_pending"}
    base = {"schema_version": MANIFEST_SCHEMA, **identity, "simulation_revision": revision,
            "source": {"plan_digest": source.get("plan_digest"),
                       "files": {prim: row["sha256"] for prim, row in sorted(source["files"].items())}},
            "upload_sha256": upload, "permission": basis, "built_at_epoch": int(now()), "archive": None,
            "delivered": None}
    candidate, outcome = directory / f".candidate-{revision[7:23]}.zip", None
    if needs_build:
        try:
            built = build_package(source, candidate, literals=secret_literals(root, source),
                                  words=capture_site_identity(root), max_bytes=max_bytes,
                                  deadline=time.monotonic() + build_seconds)
        except PackageRefused as error:
            outcome = {**base, "package": None, "refused": error.reason, "detail": error.detail,
                       "refused_at_epoch": int(now())}
        except ValueError as error:
            if str(error) not in (OVER_CAP, BUILD_TIMED_OUT):
                candidate.unlink(missing_ok=True)
                raise
            outcome = {**base, "package": None, "refused": str(error), "refused_at_epoch": int(now())}
    try:
        with package_lock(root):
            if capture_withdrawn(root):
                return {"status": "withdrawn"}
            # The whole decision again before any upload: a takedown, a recorded "no" or removal from Blueprint's
            # list that landed while the package was being built stops it here.
            if not permission.delivery_gate_allows(root, surface="website_company_simulation"):
                return {"status": "refused", "reason": "website_company_simulation_takedown_gate_closed"}
            if not permission.basis_holds(root, identity["capture_id"], basis["permission_basis"],
                                          upload_sha256(root), now(), _owned_list()):
                recorded = permission.read_check(root)
                if recorded.get("outcome") == "rights_refused":
                    return {"status": "refused",
                            "reason": _own_reason(recorded.get("reason") or permission.RIGHTS_REFUSED)}
                return {"status": "not_permitted"}
            if needs_build:
                fresh = simulation_source(root, queue_root=queue_root)
                if fresh is None or simulation_revision(fresh, max_bytes=max_bytes) != revision:
                    return {"status": "failed", "reason": "website_company_simulation_changed_during_build"}
                if outcome is not None:
                    package_path.unlink(missing_ok=True)
                    write_json(manifest_path, outcome)
                    logger.warning("website_company_simulation_not_built capture=%s reason=%s",
                                   identity["capture_id"], outcome["refused"])
                    if outcome["refused"] == OVER_CAP:
                        return {"status": "over_size_cap"}
                    if outcome["refused"] == BUILD_TIMED_OUT:
                        return {"status": "build_timed_out"}
                    return {"status": "refused", "reason": outcome["refused"], "detail": outcome["detail"]}
                os.replace(candidate, package_path)
                manifest = {**base, "package": {**built, "path": PACKAGE_FILE},
                            **({"refused_by_webapp": manifest["refused_by_webapp"]}
                               if isinstance(manifest.get("refused_by_webapp"), Mapping) else {})}
            return _publish_built(root, identity, manifest, manifest_path, package_path, basis, upload, built,
                                  revision, check, now, archive, retract, deliver)
    finally:
        candidate.unlink(missing_ok=True)


def _publish_built(root, identity, manifest, manifest_path, package_path, basis, upload, built, revision, check,
                   now, archive, retract, deliver):
    manifest["permission"], manifest["upload_sha256"] = basis, upload
    record = package_record(identity=identity, manifest=manifest)
    manifest["simulation_digest"] = record["simulation_digest"]
    write_json(manifest_path, manifest)
    stored = manifest.get("archive") or {}
    if stored.get("digest") != built["digest"] or permission.hidden(stored):
        manifest["archive"] = archive(package_path, identity=identity)
        if manifest["archive"]["digest"] != built["digest"]:
            raise ValueError("website_company_simulation_archive_mismatch")
        write_json(manifest_path, manifest)
    if capture_withdrawn(root):
        permission.take_back(manifest, manifest_path, retract)
        return {"status": "withdrawn"}
    try:
        deliver(record, identity=identity)
    except Exception as error:  # noqa: BLE001 - classified below; the archive stands only with rights
        reason = _own_reason(error)
        if permission.RIGHTS_REFUSALS.search(str(error)):
            permission.take_back(manifest, manifest_path, retract)
            manifest["refused_by_webapp"] = {"simulation_revision": revision, "reason": reason,
                                             "at_epoch": int(now()), "rights_fingerprint": check.get("rights_fingerprint")}
            write_json(manifest_path, manifest)
            return {"status": "refused", "reason": reason}
        manifest["last_delivery_error"] = reason
        write_json(manifest_path, manifest)
        return {"status": "failed", "reason": reason}
    manifest["delivered"] = {"simulation_digest": record["simulation_digest"], "at_epoch": int(now())}
    manifest.pop("last_delivery_error", None)
    manifest.pop("refused_by_webapp", None)
    write_json(manifest_path, manifest)
    return {"status": "published", "simulation_revision": revision, "simulation_digest": record["simulation_digest"]}


def tick(storage_root: Path, *, stop_reason: Callable[[], str | None] | None = None,
         budget_seconds: float = TICK_BUDGET_SECONDS, clock: Callable[[], float] = time.monotonic,
         **options: Any) -> dict[str, int]:
    """One pass over every capture the clip pass covers; counts of each outcome.

    As the clip's pass: between captures it stops for a hold, a quiesce or a new release (``stop_reason``) and
    starts no capture after ``budget_seconds``; the captures left are counted ``deferred``. One build also has
    its own deadline (``BUILD_DEADLINE_SECONDS``).
    """
    counts: dict[str, int] = {}
    started = clock()
    captures = clip.capture_roots(storage_root)
    for index, capture in enumerate(captures):
        reason = stop_reason() if stop_reason is not None else None
        if reason or clock() - started >= budget_seconds:
            counts["deferred"] = len(captures) - index
            counts["stopped_" + (reason or "time_budget")] = 1
            break
        status = publish_company_simulation(capture, **options)["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


#: The route's clock (the pass takes ``now`` from its caller); a test can set it.
def _wall_clock() -> float:
    return time.time()


def open_company_simulation(*, request_id: str, capture_id: str, digest: str, storage_root: Path | None,
                            range_header: str | None = None, client: Any = None, bucket: str | None = None,
                            now: float | None = None) -> tuple[Any, int, tuple[int, int] | None]:
    """The archived package body, its full size and the range served, for a read the WebApp authorized.

    It fails closed: with no storage root, a withdrawn site (found without the workspace), no capture
    workspace, a closed takedown gate, a manifest that does not name this digest as archived and not taken back,
    or a basis that no longer holds at ``now`` (Blueprint's list names this upload, or the last live grant is
    younger than the bound and nothing has said no since), nothing is served.
    """
    if (storage_root is None or not capture_name_matches_request(request_id, capture_id)
            or not _DIGEST.fullmatch(str(digest)) or permission.site_withdrawn(storage_root, request_id)):
        raise CompanySimulationUnavailable("website_company_simulation_unavailable")
    roots = [root for root in Path(storage_root).glob(f"*/scenes/site-{request_id}/captures/{capture_id}")
             if root.is_dir() and not root.is_symlink()]
    if len(roots) != 1:
        raise CompanySimulationUnavailable("website_company_simulation_unavailable")
    manifest = _read(roots[0].joinpath(*DIRECTORY) / MANIFEST_FILE)
    stored = manifest.get("archive") or {}
    moment = _wall_clock() if now is None else now
    if (not permission.delivery_gate_allows(roots[0], surface="website_company_simulation")
            or manifest.get("schema_version") != MANIFEST_SCHEMA or stored.get("digest") != digest
            or permission.hidden(stored)
            or not permission.basis_still_holds(roots[0], capture_id, manifest, moment, _owned_list())):
        raise CompanySimulationUnavailable("website_company_simulation_unavailable")
    from .task_evaluation_configured_scene_object_keys import LARGE_ARTIFACT_KEY_PREFIX
    from .task_evaluation_configured_scene_object_store import _artifact_object_store_client, _object_missing
    if client is None or bucket is None:
        client, bucket = _artifact_object_store_client()
    key = f"{LARGE_ARTIFACT_KEY_PREFIX}/{ARTIFACT_KIND}/sha256/{digest[7:]}/{package_filename(capture_id, request_id)}"
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception as error:  # noqa: BLE001 - provider exception shapes vary
        if _object_missing(error):
            raise CompanySimulationUnavailable("website_company_simulation_unavailable") from None
        raise
    size = int(head["ContentLength"])
    if (head.get("Metadata") or {}).get("sha256") != digest[7:] or not 0 < size <= MAX_PACKAGE_BYTES:
        raise CompanySimulationUnavailable("website_company_simulation_unavailable")
    served = clip.parse_range(range_header, size)
    extra = {"Range": f"bytes={served[0]}-{served[1]}"} if served else {}
    return client.get_object(Bucket=bucket, Key=key, **extra)["Body"], size, served


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("publish", help="make and publish one capture's simulation package if it is not current")
    one.add_argument("--capture-root", type=Path, required=True)
    every = commands.add_parser("tick", help="make and publish every capture's simulation package that is not current")
    every.add_argument("--storage-root", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    result = (publish_company_simulation(args.capture_root) if args.command == "publish"
              else tick(args.storage_root))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
