"""Which provider writes one job's ARTISAN text: chosen once per grant, on the host, before preparation.

Anthropic only when all of these hold (otherwise OpenAI, with the reason):

1. the website's grant allows both providers (``authoring_providers`` is ``openai`` and
   ``anthropic``, with a well-formed ``anthropic_provider_terms_reference``);
2. the host prefers Anthropic (``BLUEPRINT_ARTISAN_PREFER_ANTHROPIC=1``);
3. ``ANTHROPIC_API_KEY_FILE`` is present and only its owner can read it;
4. the Pipeline's own privacy rule: the job's Privacy version is ``PRIVACY_POLICY_VERSION``, or
   the version is older or absent and the job carries ``TYPED_CONSENT``;
5. the existing unbilled model read (``claude_opus_model_preflight``) answers for the key and model.

No paid request is made to choose; funds are found by the first real call. Each attempt and
renewal re-reads conditions 2 and 3. An attempt that ends on Anthropic's no-funds refusal, or one
compiled with the setting off or the key file gone or no longer owner-only, leaves a sealed OpenAI
successor keyed by the grant, with its own reason: the handoff and every later attempt and renewal
of that grant use it, so a job moved to OpenAI stays there. Only a new grant chooses again.
"""
from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

SCHEMA_VERSION = "artisan_provider_choice.v1"
PREFER_ANTHROPIC_ENV = "BLUEPRINT_ARTISAN_PREFER_ANTHROPIC"
KEY_FILE_ENV = "ANTHROPIC_API_KEY_FILE"
#: The Privacy version under which a site's job may go to Anthropic without a separate consent.
PRIVACY_POLICY_VERSION = "2026-11-13.1"
#: The typed consent an older or unversioned job must carry instead.
TYPED_CONSENT = {"statement_version": "2026-09-24.v1", "provider": "anthropic", "model": "claude-opus-5-5"}
BOTH_PROVIDERS = frozenset({"openai", "anthropic"})
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
#: ARTISAN's text roles on Anthropic; the design-board image models stay those of ``ARTISAN_MODELS``.
ARTISAN_CLAUDE_MODELS = {"provider": "anthropic", "model": "claude-opus-5-5", "thinking": "max",
                         "image_model": "gpt-6.1-sol", "image_tool_model": "gpt-image-2.5-sunburst"}
MODEL_READ_PASSED = "model_metadata_available_no_inference"
CHOICE_FILE = "artisan_provider_choice.json"
SUCCESSOR_ROOT_ENV = "BLUEPRINT_ARTISAN_PROVIDER_CHOICE_ROOT"
#: Used only where the control-plane state directory exists (the host), as the paid-spend meter's root.
DEFAULT_SUCCESSOR_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/artisan-provider-choice")
NO_FUNDS_REASON = "anthropic_no_funds_ended_attempt"
SETTING_OFF_REASON = "host_setting_off_after_choice"
KEY_UNUSABLE_REASON = "anthropic_key_file_unusable_after_choice"
REBIND_REASONS = frozenset({NO_FUNDS_REASON, SETTING_OFF_REASON, KEY_UNUSABLE_REASON})


def both_providers_granted(grant: Mapping[str, Any]) -> bool:
    providers = grant.get("authoring_providers")
    return (isinstance(providers, list) and len(providers) == 2 and set(providers) == BOTH_PROVIDERS
            and DIGEST.fullmatch(str(grant.get("anthropic_provider_terms_reference") or "")) is not None)


def privacy_refusal(grant: Mapping[str, Any]) -> str | None:
    """Why the Pipeline's own rule keeps this grant off Anthropic, or ``None``."""
    version = grant.get("privacy_policy_version")
    if version is not None and (not isinstance(version, str) or not version.strip()):
        return "privacy_policy_version_invalid"
    if version == PRIVACY_POLICY_VERSION:
        return None
    consents = grant.get("site_consents", [])
    if not isinstance(consents, list) or not all(isinstance(item, Mapping) for item in consents):
        return "site_consents_invalid"
    if any(dict(item) == TYPED_CONSENT for item in consents):
        return None
    return "privacy_version_not_listed_and_typed_consent_absent"


def owner_only_key_file(environ: Mapping[str, str]) -> Path | None:
    """``ANTHROPIC_API_KEY_FILE`` when it names an absolute, regular, non-empty file only its owner (us) reads."""
    path = Path(named := str(environ.get(KEY_FILE_ENV) or "").strip())
    try:
        metadata = os.stat(path, follow_symlinks=False) if named and path.is_absolute() else None
    except OSError:
        return None
    mode = stat.S_IMODE(metadata.st_mode) if metadata is not None else 0
    if (metadata is None or not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid()
            or mode & 0o077 or not mode & 0o400 or not 0 < metadata.st_size <= 16_384):
        return None
    return path


def model_read() -> dict[str, Any]:
    """The existing unbilled ``GET /v1/models/{model}`` for the configured key (no inference)."""
    from .claude_opus_authoring_invoker import ClaudeAuthoringBlocked
    from .claude_opus_model_preflight import preflight
    try:
        value = preflight()
    except ClaudeAuthoringBlocked as exc:
        return {"status": str(exc)}
    return {"status": value["status"], "model_metadata_digest": value["model_metadata_digest"]}


def _sealed(record: dict[str, Any]) -> dict[str, Any]:
    record["digest"] = canonical_digest(record, digest_field="digest")
    return record


def choose_authoring_provider(*, grant: Mapping[str, Any], environ: Mapping[str, str] | None = None,
                              read: Callable[[], Mapping[str, Any]] = model_read) -> dict[str, Any]:
    values = os.environ if environ is None else environ
    record: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "provider": "openai", "reason": None,
                              "grant_authority_digest": grant.get("authority_digest"), "model_read": None}
    if not both_providers_granted(grant):
        record["reason"] = "grant_allows_openai_only"
    elif str(values.get(PREFER_ANTHROPIC_ENV) or "") != "1":
        record["reason"] = "host_prefers_openai"
    elif owner_only_key_file(values) is None:
        record["reason"] = "anthropic_key_file_absent_or_not_owner_only"
    elif (refusal := privacy_refusal(grant)) is not None:
        record["reason"] = refusal
    elif (answer := dict(read()))["status"] != MODEL_READ_PASSED:
        record.update(reason="model_read:" + str(answer["status"]), model_read=answer)
    else:
        record.update(provider="anthropic", reason="all_conditions_met", model_read=answer,
                      anthropic_provider_terms_reference=grant["anthropic_provider_terms_reference"])
    return _sealed(record)


def validate_choice(record: Any) -> dict[str, Any]:
    """The sealed choice, re-checked wherever it is read; raises ``ValueError`` otherwise."""
    if (not isinstance(record, Mapping) or record.get("schema_version") != SCHEMA_VERSION
            or record.get("digest") != canonical_digest(record, digest_field="digest")
            or record.get("provider") not in BOTH_PROVIDERS
            or (record["provider"] == "anthropic" and (
                record.get("reason") != "all_conditions_met"
                or (record.get("model_read") or {}).get("status") != MODEL_READ_PASSED
                or DIGEST.fullmatch(str(record.get("anthropic_provider_terms_reference") or "")) is None))):
        raise ValueError("artisan_provider_choice_invalid")
    return dict(record)


def choice_for_grant(record: Any, *, grant: Mapping[str, Any], backend: str) -> dict[str, Any] | None:
    """The sealed choice re-checked against the grant and backend it is used with (``None`` stays ``None``)."""
    if record is None:
        return None
    value = validate_choice(record)
    from .task_object_authoring_backend import ARTISAN_BACKEND
    if (backend != ARTISAN_BACKEND or not both_providers_granted(grant)
            or value.get("grant_authority_digest") != grant.get("authority_digest")
            # A stored Anthropic choice is held to the privacy rule again, for the grant in hand.
            or (value["provider"] == "anthropic" and privacy_refusal(grant) is not None)):
        raise ValueError("website_artisan_provider_choice_not_for_this_grant")
    return value


def _successor_path(grant_digest: Any, environ: Mapping[str, str] | None) -> Path | None:
    configured = str((os.environ if environ is None else environ).get(SUCCESSOR_ROOT_ENV) or "").strip()
    root = Path(configured) if configured else (
        DEFAULT_SUCCESSOR_ROOT if DEFAULT_SUCCESSOR_ROOT.parent.is_dir() else None)
    text = str(grant_digest or "")
    return root / (text[7:] + ".json") if root is not None and DIGEST.fullmatch(text) else None


def retained_successor(grant_digest: Any, *, environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """The grant's sealed OpenAI successor (for the rest of the grant), or ``None``."""
    path = _successor_path(grant_digest, environ)
    if path is None or not path.is_file() or path.is_symlink():
        return None
    try:
        record = validate_choice(json.loads(path.read_text()))
    except (ValueError, OSError):
        return None
    if (record.get("reason") not in REBIND_REASONS or record["provider"] != "openai"
            or record.get("grant_authority_digest") != grant_digest):
        return None
    return record


def _save_successor(choice: Mapping[str, Any], reason: str, environ: Mapping[str, str] | None,
                    **extra: str) -> dict[str, Any]:
    """The grant's sealed OpenAI successor of its Anthropic ``choice``, saved once; the first one saved stays."""
    record = _sealed({"schema_version": SCHEMA_VERSION, "provider": "openai", "reason": reason,
                      "grant_authority_digest": choice["grant_authority_digest"], "model_read": None, **extra,
                      "supersedes": {"digest": choice["digest"], "provider": "anthropic"}})
    path = _successor_path(choice.get("grant_authority_digest"), environ)
    if path is None:
        return record  # nowhere to keep it (off the host): this attempt only
    existing = retained_successor(choice["grant_authority_digest"], environ=environ)
    if existing is not None:
        return existing if existing["supersedes"].get("digest") == choice["digest"] else record
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(record, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(raw, path)
    finally:
        Path(raw).unlink(missing_ok=True)
    return record


def record_no_funds_successor(predecessor: Any, *, run_id: str,
                              environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """After an Anthropic attempt ended for lack of funds: the grant's choice becomes OpenAI, sealed, once."""
    choice = validate_choice(predecessor)
    if choice["provider"] != "anthropic" or _successor_path(choice.get("grant_authority_digest"), environ) is None:
        return None
    return _save_successor(choice, NO_FUNDS_REASON, environ, ended_attempt_run_id=str(run_id))


def _rebound(configuration: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    from .task_object_authoring_backend import ARTISAN_MODELS
    return {**configuration, "authoring_models": dict(ARTISAN_MODELS), "artisan_provider_choice": dict(record)}


def with_provider_successor(configuration: dict[str, Any], *,
                            environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """A stage 3 configuration chosen for Anthropic, rebound to OpenAI when its grant has a successor,
    or (saving one) when the host's setting is off or the key file is gone or no longer owner-only."""
    choice = configuration.get("artisan_provider_choice")
    if not isinstance(choice, Mapping) or choice.get("provider") != "anthropic":
        return configuration
    grant_digest = validate_choice(choice).get("grant_authority_digest")
    successor = retained_successor(grant_digest, environ=environ)
    if successor is not None and successor["supersedes"].get("digest") == choice["digest"]:
        return _rebound(configuration, successor)
    values = os.environ if environ is None else environ
    reason = (SETTING_OFF_REASON if str(values.get(PREFER_ANTHROPIC_ENV) or "") != "1"
              else KEY_UNUSABLE_REASON if owner_only_key_file(values) is None else None)
    if reason is None:
        return configuration
    return _rebound(configuration, _save_successor(choice, reason, environ))


def as_rebound(stored: Mapping[str, Any], derived: dict[str, Any]) -> dict[str, Any]:
    """``derived`` rebound as ``stored`` was, when ``stored`` carries a sealed rebind of ``derived``'s
    Anthropic choice; else ``derived``. Re-checks a frozen attempt without re-reading the host."""
    choice, record = derived.get("artisan_provider_choice"), stored.get("artisan_provider_choice")
    if not isinstance(choice, Mapping) or choice.get("provider") != "anthropic":
        return derived
    try:
        value = validate_choice(record)
    except ValueError:
        return derived
    if (value["provider"] != "openai" or value.get("reason") not in REBIND_REASONS
            or value.get("supersedes") != {"digest": choice.get("digest"), "provider": "anthropic"}):
        return derived
    return _rebound(derived, value)


def retained_or_new_choice(*, root: Path, grant: Mapping[str, Any], environ: Mapping[str, str] | None = None,
                           read: Callable[[], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The grant's choice: its no-funds successor, else the one recorded beside this preparation, else a new one.

    A new record is written, through the handoff's lease-fenced writer, before it is returned.
    """
    from .core.handoff_lease import guarded_handoff_write_json
    successor = retained_successor(grant.get("authority_digest"), environ=environ)
    if successor is not None:
        return successor
    path = root / CHOICE_FILE
    if path.is_file() and not path.is_symlink():
        try:
            retained = validate_choice(json.loads(path.read_text()))
        except (ValueError, OSError):
            retained = None
        if retained is not None and retained.get("grant_authority_digest") == grant.get("authority_digest"):
            return retained
    record = choose_authoring_provider(grant=grant, environ=environ, read=read or model_read)
    guarded_handoff_write_json(path, record)
    return record


def artisan_text_provider(configuration: Mapping[str, Any]) -> str:
    """``anthropic`` only for an ARTISAN configuration whose sealed choice chose it; else ``openai``."""
    choice = configuration.get("artisan_provider_choice")
    claude = configuration.get("authoring_models") == ARTISAN_CLAUDE_MODELS
    provider = "openai" if choice is None else validate_choice(choice)["provider"]
    if claude != (provider == "anthropic"):
        raise ValueError("artisan_provider_choice_models_mismatch")
    return provider
