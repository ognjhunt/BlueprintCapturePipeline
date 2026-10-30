"""Current-source candidate preparation notes; never a screening plan or result.

Uses the effective catalog and operator registration snapshot without a scene,
credentials, scored salt, quotes, reservations, grants or execution descriptors.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .website_preparation_status import STATUS_V4, read_preparation_status, validate_selectors

SCHEMA = "website_candidate_readiness.v1"
MAX_BYTES = 256 * 1024
MAX_CANDIDATES = 256  # Same bound as the effective catalog; never truncate rows.
MAX_REASONS = 32
PATH = "/api/live-pipeline/website-candidate-readiness"
_CODE = re.compile(r"[a-z][a-z0-9_]{1,127}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class CandidateReadinessUnavailable(ValueError):
    """Fixed public codes, never raw host errors or configuration values."""


def _require(value, reason="website_candidate_readiness_unavailable"):
    if not value:
        raise CandidateReadinessUnavailable(reason)


def _profile_snapshot(root: Path, selected: Mapping[str, str]) -> tuple[dict, dict]:
    """Same catalog settings as a real profile, without minting a scene or reading salt.

    Before compilation, defaults describe installed registrations only. A retained
    derived row adds source identity; it does not turn this view into compatibility.
    """
    from . import team_candidate_quick10_registry as registry

    value = registry.load_registry()
    if value is None:
        return {}, {
            "registry_digest": None,
            "profile_digest": digest({}),
            "profile_source": "not_configured",
        }
    capture = str(root.resolve())
    matches = [
        dict(row)
        for row in value["profiles"]
        if isinstance(row, Mapping) and row.get("capture_root") == capture
    ]
    _require(len(matches) <= 1)
    profile, origin = {}, "not_configured"
    if matches:
        profile, origin = matches[0], "capture_profile"
    else:
        settings = registry._derived_settings(value)
        if settings is not None:
            profile, origin = dict(settings["defaults"]), "operator_defaults"
            path = registry._derived_path(settings["root"], capture)
            if path.exists() or path.is_symlink():
                row = registry._read(
                    registry._operator_file(
                        path, code="team_candidate_derived_profile_not_operator_owned"
                    )
                )
                _require(
                    row.get("schema_version") == registry.DERIVED_SCHEMA
                    and not set(row)
                    - {*registry.DERIVED_FIELDS, "schema_version", "profile_digest"}
                    and row.get("capture_root") == capture
                    and row.get("profile_digest")
                    == registry.canonical_digest(row, digest_field="profile_digest")
                )
                profile = {
                    **{key: item for key, item in row.items() if key in registry.DERIVED_FIELDS},
                    **profile,
                }
                origin = "derived_profile"
    # A scene-bound registration is considered only when its retained scene
    # identity actually exists. Defaults alone cannot fabricate that authority.
    if origin in {"capture_profile", "derived_profile"}:
        _require(profile.get("scene_id") in (None, selected["scene_id"]))
        profile = registry._site_screening_profile(value, profile)
    configured = {"screening_candidates": profile.get("screening_candidates") or {}}
    return configured, {
        "registry_digest": value["registry_digest"],
        "profile_digest": digest({"source": origin, "profile": profile}),
        "profile_source": origin,
    }


def _public_name(row: Mapping) -> str:
    name = row["display_name"].strip()
    # These fields are labels, never endpoints, contact details or host paths.
    if (
        any(ord(c) < 32 for c in name)
        or "://" in name
        or "@" in name
        or "\\" in name
        or "=" in name
        or re.search(r"(?:^|\s)/[A-Za-z0-9_.-]", name)
        or re.search(r"(?i)\b(bearer|api[_ -]?key|password|secret|token)\b", name)
    ):
        return row["candidate_id"].replace("_", " ")
    return name.replace("_", " ")[:160]


def _candidate(row: Mapping) -> dict:
    _require(_ID.fullmatch(row["candidate_id"]) is not None)
    reasons = list(row["eligibility"]["blockers"])
    if row["eligibility"]["commercial_use"] not in {"allowed", "allowed_with_conditions"}:
        reasons.append("commercial_use_not_admitted")
    # Credential environment variables are not inspected here. A registered
    # hosted route's effective catalog already records file-access failures;
    # absence of such a finding is not evidence of a successful account call.
    reasons = sorted(
        set(
            reason
            if isinstance(reason, str) and _CODE.fullmatch(reason)
            else "candidate_configuration_needs_review"
            for reason in reasons
        )
    )
    _require(len(reasons) <= MAX_REASONS, "website_candidate_readiness_bound_exceeded")
    return {
        "candidate_id": row["candidate_id"],
        "display_name": _public_name(row),
        "configuration_digest": row["configuration_digest"],
        "policy_kind": row["policy"]["kind"],
        "compatibility": "not_evaluated",
        "known_reasons": reasons,
    }


def read_candidate_readiness(*, capture_root: Path, selectors: Mapping[str, str]) -> dict:
    """A fresh read is source/registry-bound, and has no writes or launch authority."""
    from .beta_policy_screening_team import catalog_for_profile

    selected = validate_selectors(dict(selectors))
    before = read_preparation_status(
        capture_root=capture_root, selectors=selected, schema_version=STATUS_V4
    )
    try:
        profile, snapshot = _profile_snapshot(capture_root, selected)
        catalog = catalog_for_profile(profile)
        _require(
            0 < len(catalog["candidates"]) <= MAX_CANDIDATES,
            "website_candidate_readiness_bound_exceeded",
        )
        rows = [_candidate(row) for row in catalog["candidates"]]
        _require(len({row["candidate_id"] for row in rows}) == len(rows))
        result = {
            "schema_version": SCHEMA,
            **selected,
            "claim_ceiling": "candidate_preparation_only",
            "compatibility": "not_evaluated",
            "preparation_revision": before["revision"],
            "preparation_status_digest": before["status_digest"],
            "catalog_digest": catalog["catalog_digest"],
            **snapshot,
            "candidate_count": len(rows),
            "candidates": rows,
        }
        result["readiness_digest"] = digest(result)
        _require(
            len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode())
            <= MAX_BYTES,
            "website_candidate_readiness_bound_exceeded",
        )
        # A mutation during source/model verification must not produce a mixed
        # snapshot. No grants, catalog records or preparation ledgers are written.
        after_profile, after_snapshot = _profile_snapshot(capture_root, selected)
        _require(after_profile == profile and after_snapshot == snapshot)
        _require(catalog_for_profile(after_profile)["catalog_digest"] == catalog["catalog_digest"])
        after = read_preparation_status(
            capture_root=capture_root, selectors=selected, schema_version=STATUS_V4
        )
        _require(before == after)
        return result
    except CandidateReadinessUnavailable:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CandidateReadinessUnavailable("website_candidate_readiness_unavailable") from exc
