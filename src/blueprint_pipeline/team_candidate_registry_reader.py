"""Read the operator team-candidate registry without the scene or launch lanes.

The single implementation of how the operator registry
(``BLUEPRINT_TEAM_CANDIDATE_QUICK10_REGISTRY``) and its derived-profile settings
are read and owner-checked, and of the task-bound G1 overlay a profile receives.
``team_candidate_quick10_registry`` builds scene facts, salts and offers on top
of it. This module's import graph stays free of scene compilation, providers,
GPU runtimes, the paid-resource allocator and screening launch, so a CPU intake
service can read registrations (``tests/test_live_pipeline_import_isolation.py``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .beta_policy_screening_g1_catalog import CANDIDATES as G1_CANDIDATES
from .decision_evidence_contracts import canonical_digest
from .native_g1_beta_authority import SOURCE_FIELDS as G1_SOURCE_FIELDS

REGISTRY_ENV = "BLUEPRINT_TEAM_CANDIDATE_QUICK10_REGISTRY"
REGISTRY_SCHEMA = "blueprint.team_candidate_quick10_registry.v1"
DERIVED_SCHEMA = "blueprint.team_candidate_derived_profile.v1"
# What a prepared scene's own records determine (the first six are required, plus one scene half:
# the scene's own compiled packet, or a frozen internal Quick-10 pair).
DERIVED_FIELDS = ("capture_root", "scene_id", "task_id", "scenario_id", "robot_free_probe_root",
                  "scene_revision_digest", "team_scene_inputs_path", "pair_runtime_inputs_path",
                  "pair_activation_path", "nominal_dynamic_friction", "family_scene_inputs_path")
REQUIRED_DERIVED_FIELDS = DERIVED_FIELDS[:6]
# What only the operator decides: who may fund a run, at what rate, and each team's rights.
OPERATOR_FIELDS = ("allowed_founder_approvers", "max_hourly_rate_usd", "open_loop_horizon", "team_rights",
                   "machine_avoidlist_path", "private_robot_models", "prepared_robot_models",
                   "screening_candidates", "screening_hard_ttl_seconds", "screening_candidate_cap_usd",
                   "native_runtime_source_receipt", "default_robot_model", "site_task_qualification",
                   "simulator_measurement_producer")
# The service's own automatic G1 registrations (``native_g1_site_registration``, owner's standing approval
# 2026-11-06): one ``<site_id>.json`` row per website site, in the root registry's row shape.
SITE_REGISTRATION_ROOT_ENV = "BLUEPRINT_NATIVE_G1_SITE_REGISTRATION_ROOT"
DEFAULT_SITE_REGISTRATION_ROOT = "/var/lib/blueprint/pipeline-control-plane/native-g1-site-registrations"
AUTO_REGISTRATION_ENV = "BLUEPRINT_NATIVE_G1_AUTO_REGISTRATION"
WEBSITE_SITE_ID = re.compile(r"site-[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")


class TeamCandidateRegistryError(ValueError):
    """The operator registry or a profile it names is unusable."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TeamCandidateRegistryError("team_candidate_registry_record_invalid")
    return value


def load_registry() -> dict[str, Any] | None:
    raw = os.environ.get(REGISTRY_ENV)
    if not raw:
        return None
    path = Path(raw)
    try:
        status = path.lstat()
    except OSError as exc:
        raise TeamCandidateRegistryError("team_candidate_registry_missing") from exc
    if (not path.is_absolute() or stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode)
            or status.st_mode & 0o022 or status.st_uid not in {0, os.getuid()}):
        raise TeamCandidateRegistryError("team_candidate_registry_not_operator_owned")
    registry = _read(path)
    if (registry.get("schema_version") != REGISTRY_SCHEMA
            or registry.get("registry_digest") != canonical_digest(registry, digest_field="registry_digest")
            or not isinstance(registry.get("profiles"), list)):
        raise TeamCandidateRegistryError("team_candidate_registry_binding_invalid")
    return registry


def _operator_file(path: Path, *, code: str) -> Path:
    try:
        status = path.lstat()
    except OSError as exc:
        raise TeamCandidateRegistryError(code) from exc
    if (not path.is_absolute() or stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode)
            or status.st_mode & 0o022 or status.st_uid not in {0, os.getuid()}):
        raise TeamCandidateRegistryError(code)
    return path


def _derived_settings(registry: Mapping[str, Any]) -> dict[str, Any] | None:
    settings = registry.get("derived_profiles")
    if settings is None:
        return None
    root = Path(str((settings or {}).get("root") or ""))
    defaults = (settings or {}).get("defaults")
    if (not isinstance(settings, Mapping) or not root.is_absolute() or not isinstance(defaults, Mapping)
            or set(defaults) - set(OPERATOR_FIELDS) or set(OPERATOR_FIELDS[:3]) - set(defaults)):
        raise TeamCandidateRegistryError("team_candidate_derived_profile_settings_invalid")
    return {"root": root, "defaults": dict(defaults), "salt_key_path": settings.get("scored_matrix_salt_key_path")}


def _derived_path(root: Path, capture_root: str) -> Path:
    return root / (hashlib.sha256(capture_root.encode()).hexdigest()[:32] + ".json")


def auto_registration_enabled() -> bool:
    """On by default (the owner's standing approval, 2026-11-06); ``0``, false, off or no turns it off."""
    return os.environ.get(AUTO_REGISTRATION_ENV, "1").strip().lower() not in {"0", "false", "off", "no"}


def site_registration_path(site_id: Any, root: str | Path | None = None) -> Path | None:
    """``<root>/<site_id>.json`` for a website site id, else None (never a path outside the root)."""
    if not isinstance(site_id, str) or not WEBSITE_SITE_ID.fullmatch(site_id) or site_id.endswith(".status"):
        return None
    return Path(root or os.environ.get(SITE_REGISTRATION_ROOT_ENV) or DEFAULT_SITE_REGISTRATION_ROOT) / (site_id + ".json")


def _site_row_valid(row: Any) -> bool:
    return (isinstance(row, Mapping) and set(row) == {"source", "screening_candidates"}
            and isinstance(row["source"], Mapping) and set(row["source"]) == set(G1_SOURCE_FIELDS)
            and isinstance(row["screening_candidates"], Mapping)
            and not set(row["screening_candidates"]) - set(G1_CANDIDATES)
            and all(isinstance(item, Mapping) and set(item) == {"native_g1"}
                    for item in row["screening_candidates"].values()))


def site_registration_overlay(site_id: Any, root: str | Path | None = None) -> dict[str, Any] | None:
    """The service's own row for one site, or None.

    Only a regular, non-symlink file directly under an absolute, non-symlink root, owned by root or this
    service user and writable by no one else, in exactly the root registry's row shape. Anything else is
    ignored, never raised: a G1 registration problem must not stop the site's other candidates.
    """
    path = site_registration_path(site_id, root)
    if path is None or not auto_registration_enabled():
        return None
    try:
        if not path.is_absolute() or any(part.is_symlink() for part in path.parents):
            return None
        status = path.lstat()
        if (stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode) or status.st_mode & 0o022
                or status.st_uid not in {0, os.getuid()}):
            return None
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return dict(row) if _site_row_valid(row) else None


def _source_current(source: Mapping[str, Any], profile: Mapping[str, Any]) -> bool:
    context = _read(Path(profile["capture_root"]) / "pipeline/website_task_context.json")
    return (source["task_id"] == profile.get("task_id")
            and source["scene_revision_digest"] == profile.get("scene_revision_digest")
            and source["capture_id"] == context.get("capture_id")
            and source["task_context_digest"] == context.get("context_digest"))


def _site_screening_profile(registry: Mapping[str, Any], profile: Mapping[str, Any], *,
                            overlay_root: str | Path | None = None) -> dict[str, Any]:
    """Overlay task-bound G1 receipts without freezing the derived scene facts.

    The root-owned registry's row wins while it names the current source. Otherwise the service's
    automatic registration for the site (``site_registration_overlay``) is read with the same checks.
    """
    value = dict(profile)
    row = (registry.get("site_screening_candidates") or {}).get(profile.get("scene_id"))
    if row is not None and not _site_row_valid(row):
        raise TeamCandidateRegistryError("team_candidate_site_screening_registration_invalid")
    if row is None or not _source_current(row["source"], profile):
        try:
            row = site_registration_overlay(profile.get("scene_id"), overlay_root)
            if row is not None and not _source_current(row["source"], profile):
                row = None
        except (OSError, ValueError, KeyError, TypeError):
            row = None
    if row is None:
        return value
    value["screening_candidates"] = {**value.get("screening_candidates", {}), **row["screening_candidates"]}
    value["native_g1_registration_source"] = dict(row["source"])
    return value


__all__ = [
    "AUTO_REGISTRATION_ENV",
    "DEFAULT_SITE_REGISTRATION_ROOT",
    "DERIVED_FIELDS",
    "DERIVED_SCHEMA",
    "OPERATOR_FIELDS",
    "REGISTRY_ENV",
    "REGISTRY_SCHEMA",
    "REQUIRED_DERIVED_FIELDS",
    "SITE_REGISTRATION_ROOT_ENV",
    "TeamCandidateRegistryError",
    "auto_registration_enabled",
    "load_registry",
    "site_registration_overlay",
    "site_registration_path",
]
