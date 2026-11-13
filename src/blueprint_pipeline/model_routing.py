"""One place that says which provider and model each model-calling role uses.

``config/model_routing.v1.json`` maps every role to an ordered list of
``(provider, model, effort)``. The first entry the privacy gate permits is the
role's default; a later entry is tried only after the provider before it
answered with a typed "no funds" error (``model_failover``). The active profile
is the one setting: ``current`` is what runs today, ``same_level_claude`` is
available but not active.

Privacy is a hard stop. Every role declares the class of data it sends, and
each data class names the providers a Privacy Policy version permits for it. A
job is gated by the version it was submitted under; a version the table does not
list, or a job with none, gets ``baseline`` (today's wording). A provider not
permitted for the role's class is never in the resolved route: not as the
default and not as a failover target. A provider the policy permits only when
the site chose it explicitly is admitted only with that choice.

A route is resolved once for an attempt and then frozen: ``ResolvedRoute.record``
carries the entries and a digest, and ``route_from_record`` restores it without
reading the configuration again, so a later configuration change never changes
a running attempt.

The price table holds providers' published list prices, each with the date it
was read and its source. It is not a cap and nothing here reserves or meters
money. Inert: no call site reads this module yet.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "blueprint.model_routing.v1"
ROUTE_SCHEMA_VERSION = "blueprint.model_route.v1"
CONFIG_FILENAME = "model_routing.v1.json"
SHIPPED_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / CONFIG_FILENAME
CURRENT_PROFILE = "current"
BASELINE_PRIVACY_POLICY = "baseline"

PROVIDERS = frozenset({"openai", "anthropic", "google"})
DATA_CLASSES = frozenset({"site_frames", "site_video", "site_job_text", "internal_text"})
CALL_KINDS = frozenset({"single_call", "agent_run"})
OUTPUTS = frozenset({"text", "image"})
#: The effort words each provider's SDK accepts: openai ``ReasoningEffort``,
#: the Claude ``output_config.effort`` levels, google-genai ``ThinkingLevel``.
#: ``None`` sends no setting, as the call site does today.
EFFORTS: Mapping[str, frozenset[str]] = {
    "openai": frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"}),
    "anthropic": frozenset({"low", "medium", "high", "xhigh", "max"}),
    "google": frozenset({"MINIMAL", "LOW", "MEDIUM", "HIGH"}),
}
#: Providers that cannot produce an image; a role whose output is an image never routes to them.
TEXT_ONLY_PROVIDERS = frozenset({"anthropic"})
_PRICE_KEYS = {
    "text": (frozenset({"input", "output"}), frozenset({"input", "output", "cache_read", "cache_write",
                                                        "cache_write_1h"})),
    "image": (frozenset({"input_text", "input_image", "output_image"}),
              frozenset({"input_text", "input_image", "output_image", "output_text"})),
}


class ModelRoutingError(ValueError):
    """The routing configuration, a role or a frozen route is invalid."""


@dataclass(frozen=True)
class RouteEntry:
    provider: str
    model: str
    effort: str | None

    def as_dict(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "effort": self.effort}


@dataclass(frozen=True)
class ResolvedRoute:
    """The ordered entries one attempt may use for one role, frozen at resolution.

    ``list_prices`` holds, per entry and in the same order, the provider's
    published price row the entry was resolved with, so a meter or an
    authorization check reads the frozen figure, never a later table.
    """

    role: str
    data_class: str
    call_kind: str
    output: str
    profile: str
    privacy_policy: str
    entries: tuple[RouteEntry, ...]
    list_prices: tuple[Mapping[str, Any], ...]
    #: Entries the privacy gate removed, each with its typed reason; never tried.
    skipped: tuple[tuple[RouteEntry, str], ...]
    site_choices: tuple[str, ...]
    config_digest: str

    @property
    def default(self) -> RouteEntry:
        return self.entries[0]

    def list_price(self, index: int) -> Mapping[str, Any]:
        return self.list_prices[index]

    def record(self) -> dict[str, Any]:
        body = {
            "schema_version": ROUTE_SCHEMA_VERSION, "role": self.role, "data_class": self.data_class,
            "call_kind": self.call_kind, "output": self.output, "profile": self.profile,
            "privacy_policy": self.privacy_policy,
            "entries": [{**entry.as_dict(), "list_price": json.loads(json.dumps(price))}
                        for entry, price in zip(self.entries, self.list_prices, strict=True)],
            "skipped": [{**entry.as_dict(), "reason": reason} for entry, reason in self.skipped],
            "site_choices": list(self.site_choices), "config_digest": self.config_digest,
        }
        return {**body, "route_digest": _digest(body)}


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ModelRoutingError(code)


def _text(value: Any, code: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()) and value == value.strip(), code)
    return value


def _entry(value: Any, *, where: str) -> RouteEntry:
    _require(isinstance(value, Mapping) and set(value) == {"provider", "model", "effort"},
             f"model_routing_entry_invalid:{where}")
    provider = _text(value["provider"], f"model_routing_provider_invalid:{where}")
    _require(provider in PROVIDERS, f"model_routing_provider_unknown:{where}:{provider}")
    model = _text(value["model"], f"model_routing_model_invalid:{where}")
    effort = value["effort"]
    _require(effort is None or (isinstance(effort, str) and effort in EFFORTS[provider]),
             f"model_routing_effort_invalid:{where}:{provider}:{effort}")
    return RouteEntry(provider, model, effort)


def _rates(value: Any, *, kind: str, where: str) -> None:
    required, allowed = _PRICE_KEYS[kind]
    _require(isinstance(value, Mapping) and required <= set(value) <= allowed,
             f"model_routing_price_fields_invalid:{where}")
    for name, rate in value.items():
        _require(isinstance(rate, (int, float)) and not isinstance(rate, bool) and math.isfinite(rate) and rate >= 0,
                 f"model_routing_price_invalid:{where}:{name}")
    _require(any(value[name] > 0 for name in required), f"model_routing_price_all_zero:{where}")


def _validate_prices(prices: Any, *, today: date) -> dict[tuple[str, str], Mapping[str, Any]]:
    _require(isinstance(prices, Mapping) and set(prices) <= PROVIDERS, "model_routing_prices_invalid")
    table: dict[tuple[str, str], Mapping[str, Any]] = {}
    for provider, models in prices.items():
        _require(isinstance(models, Mapping) and bool(models), f"model_routing_prices_invalid:{provider}")
        for model, row in models.items():
            where = f"{provider}:{model}"
            _require(isinstance(row, Mapping) and row.get("kind") in _PRICE_KEYS
                     and set(row) <= {"kind", "read_on", "source", "usd_per_million_tokens", "long_prompt"},
                     f"model_routing_price_row_invalid:{where}")
            try:
                read_on = date.fromisoformat(str(row.get("read_on")))
            except ValueError as exc:
                raise ModelRoutingError(f"model_routing_price_date_invalid:{where}") from exc
            _require(read_on <= today, f"model_routing_price_date_future:{where}")
            source = _text(row.get("source"), f"model_routing_price_source_missing:{where}")
            _require(source.startswith("https://"), f"model_routing_price_source_missing:{where}")
            _rates(row.get("usd_per_million_tokens"), kind=row["kind"], where=where)
            if "long_prompt" in row:
                long_prompt = row["long_prompt"]
                threshold = long_prompt.get("above_input_tokens") if isinstance(long_prompt, Mapping) else None
                _require(isinstance(long_prompt, Mapping) and set(long_prompt) == {"above_input_tokens",
                                                                                   "usd_per_million_tokens"}
                         and isinstance(threshold, int) and not isinstance(threshold, bool) and threshold > 0,
                         f"model_routing_price_long_prompt_invalid:{where}")
                _rates(long_prompt["usd_per_million_tokens"], kind=row["kind"], where=where + ":long_prompt")
            table[(provider, model)] = row
    return table


def validate_config(config: Any, *, today: date | None = None) -> dict[str, Any]:
    """Check the whole configuration; return it unchanged or raise ``ModelRoutingError``."""
    today = today or date.today()
    _require(isinstance(config, Mapping) and config.get("schema_version") == SCHEMA_VERSION
             and set(config) <= {"schema_version", "active_profile", "note", "privacy_policies", "roles", "profiles",
                                 "prices", "provider_surcharges"},
             "model_routing_schema_invalid")
    policies = config.get("privacy_policies")
    _require(isinstance(policies, Mapping) and BASELINE_PRIVACY_POLICY in policies, "model_routing_privacy_invalid")
    for version, policy in policies.items():
        classes = policy.get("data_classes") if isinstance(policy, Mapping) else None
        _require(isinstance(classes, Mapping) and set(classes) == DATA_CLASSES,
                 f"model_routing_data_classes_invalid:{version}")
        for name, row in classes.items():
            where = f"{version}:{name}"
            _require(isinstance(row, Mapping), f"model_routing_data_class_invalid:{where}")
            for field in ("providers", "providers_by_site_choice"):
                values = row.get(field)
                _require(isinstance(values, list) and len(set(values)) == len(values) and set(values) <= PROVIDERS,
                         f"model_routing_data_class_invalid:{where}:{field}")
            _require(not set(row["providers"]) & set(row["providers_by_site_choice"]),
                     f"model_routing_data_class_invalid:{where}:overlap")
            _text(row.get("source"), f"model_routing_data_class_source_missing:{where}")
    baseline = policies[BASELINE_PRIVACY_POLICY]["data_classes"]
    for version, policy in policies.items():
        for name, row in policy["data_classes"].items():
            # A listed version may widen what a job submitted under it permits, never narrow the baseline.
            _require(set(baseline[name]["providers"]) <= set(row["providers"])
                     and set(baseline[name]["providers_by_site_choice"])
                     <= set(row["providers"]) | set(row["providers_by_site_choice"]),
                     f"model_routing_privacy_version_narrows_baseline:{version}:{name}")
    roles = config.get("roles")
    _require(isinstance(roles, Mapping) and bool(roles), "model_routing_roles_invalid")
    for role, row in roles.items():
        _require(isinstance(row, Mapping) and row.get("data_class") in DATA_CLASSES
                 and row.get("call_kind") in CALL_KINDS and row.get("output") in OUTPUTS,
                 f"model_routing_role_invalid:{role}")
    prices = _validate_prices(config.get("prices"), today=today)
    surcharges = config.get("provider_surcharges", {})
    _require(isinstance(surcharges, Mapping) and set(surcharges) <= PROVIDERS, "model_routing_surcharges_invalid")
    for provider, row in surcharges.items():
        _require(isinstance(row, Mapping) and str(row.get("source", "")).startswith("https://")
                 and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
                         for k, v in row.items() if k not in {"read_on", "source"}),
                 f"model_routing_surcharges_invalid:{provider}")
        try:
            _require(date.fromisoformat(str(row.get("read_on"))) <= today, f"model_routing_surcharges_invalid:{provider}")
        except ValueError as exc:
            raise ModelRoutingError(f"model_routing_surcharges_invalid:{provider}") from exc
    profiles = config.get("profiles")
    _require(isinstance(profiles, Mapping) and CURRENT_PROFILE in profiles, "model_routing_profiles_invalid")
    _require(config.get("active_profile") in profiles, "model_routing_active_profile_unknown")
    for profile, table in profiles.items():
        _require(isinstance(table, Mapping) and set(table) == set(roles),
                 f"model_routing_profile_roles_mismatch:{profile}")
        for role, raw_entries in table.items():
            where = f"{profile}:{role}"
            _require(isinstance(raw_entries, list) and bool(raw_entries), f"model_routing_entries_invalid:{where}")
            entries = [_entry(value, where=where) for value in raw_entries]
            _require(len({(e.provider, e.model) for e in entries}) == len(entries),
                     f"model_routing_entries_duplicate:{where}")
            for entry in entries:
                _require((entry.provider, entry.model) in prices,
                         f"model_routing_unpriced:{where}:{entry.provider}:{entry.model}")
                _require(roles[role]["output"] != "image" or entry.provider not in TEXT_ONLY_PROVIDERS,
                         f"model_routing_image_output_unsupported:{where}:{entry.provider}")
            # Whatever the profile, the role must have a default the baseline permits with no site choice.
            permitted = baseline[roles[role]["data_class"]]["providers"]
            _require(any(entry.provider in permitted for entry in entries),
                     f"model_routing_no_permitted_default:{where}")
    return dict(config)


def load_config(path: Path | None = None, *, today: date | None = None) -> dict[str, Any]:
    target = SHIPPED_CONFIG_PATH if path is None else path
    try:
        config = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelRoutingError("model_routing_config_unreadable") from exc
    return validate_config(config, today=today)


def config_digest(config: Mapping[str, Any]) -> str:
    return _digest(config)


def privacy_policy_key(config: Mapping[str, Any], version: str | None) -> str:
    """The table key that gates a job submitted under Privacy Policy ``version``."""
    if isinstance(version, str) and version != BASELINE_PRIVACY_POLICY and version in config["privacy_policies"]:
        return version
    return BASELINE_PRIVACY_POLICY


def permitted_providers(config: Mapping[str, Any], data_class: str, *, privacy_policy_version: str | None = None,
                        site_choices: frozenset[str] = frozenset()) -> frozenset[str]:
    """The privacy gate: providers the job's Privacy Policy version permits for ``data_class``."""
    key = privacy_policy_key(config, privacy_policy_version)
    row = config["privacy_policies"][key]["data_classes"].get(data_class)
    _require(row is not None, f"model_routing_data_class_unknown:{data_class}")
    chosen = set(row["providers_by_site_choice"]) & set(site_choices)
    return frozenset(row["providers"]) | frozenset(chosen)


def resolve_route(role: str, *, config: Mapping[str, Any] | None = None, profile: str | None = None,
                  privacy_policy_version: str | None = None,
                  site_choices: frozenset[str] = frozenset()) -> ResolvedRoute:
    """Resolve and freeze ``role``'s ordered route under the privacy gate.

    ``privacy_policy_version`` is the version the job was submitted under.
    ``site_choices`` names providers the site itself explicitly chose for its
    data; it only admits a provider the data class lists under
    ``providers_by_site_choice``.
    """
    config = load_config() if config is None else validate_config(config)
    roles = config["roles"]
    _require(role in roles, f"model_routing_role_unknown:{role}")
    name = config["active_profile"] if profile is None else profile
    _require(name in config["profiles"], f"model_routing_profile_unknown:{name}")
    _require(isinstance(site_choices, frozenset) and site_choices <= PROVIDERS, "model_routing_site_choices_invalid")
    row = roles[role]
    policy = privacy_policy_key(config, privacy_policy_version)
    permitted = permitted_providers(config, row["data_class"], privacy_policy_version=policy,
                                    site_choices=site_choices)
    kept: list[RouteEntry] = []
    skipped: list[tuple[RouteEntry, str]] = []
    for raw in config["profiles"][name][role]:
        entry = _entry(raw, where=f"{name}:{role}")
        if entry.provider in permitted:
            kept.append(entry)
        else:
            skipped.append((entry, f"provider_privacy_not_permitted:{entry.provider}:{row['data_class']}"))
    _require(bool(kept), f"model_routing_no_permitted_provider:{role}")
    prices = tuple(price_row(config, entry.provider, entry.model) for entry in kept)
    return ResolvedRoute(role=role, data_class=row["data_class"], call_kind=row["call_kind"], output=row["output"],
                         profile=name, privacy_policy=policy, entries=tuple(kept), list_prices=prices,
                         skipped=tuple(skipped),
                         site_choices=tuple(sorted(site_choices)), config_digest=config_digest(config))


def route_from_record(record: Mapping[str, Any]) -> ResolvedRoute:
    """Restore a frozen route exactly as resolved; the configuration is not read."""
    _require(isinstance(record, Mapping) and record.get("schema_version") == ROUTE_SCHEMA_VERSION,
             "model_route_record_invalid")
    body = {key: value for key, value in record.items() if key != "route_digest"}
    _require(record.get("route_digest") == _digest(body), "model_route_record_digest_mismatch")
    try:
        entries = tuple(_entry({k: v for k, v in value.items() if k != "list_price"}, where="record")
                        for value in record["entries"])
        prices = tuple(value["list_price"] for value in record["entries"])
        skipped = tuple((_entry({k: v for k, v in value.items() if k != "reason"}, where="record"),
                         _text(value["reason"], "model_route_record_invalid")) for value in record["skipped"])
        route = ResolvedRoute(role=_text(record["role"], "model_route_record_invalid"),
                              data_class=record["data_class"], call_kind=record["call_kind"], output=record["output"],
                              profile=_text(record["profile"], "model_route_record_invalid"),
                              privacy_policy=_text(record["privacy_policy"], "model_route_record_invalid"),
                              entries=entries, list_prices=prices, skipped=skipped, site_choices=tuple(record["site_choices"]),
                              config_digest=_text(record["config_digest"], "model_route_record_invalid"))
    except (KeyError, TypeError, AttributeError) as exc:
        raise ModelRoutingError("model_route_record_invalid") from exc
    for index, price in enumerate(prices):
        _require(isinstance(price, Mapping) and price.get("kind") in _PRICE_KEYS, "model_route_record_invalid")
        _rates(price.get("usd_per_million_tokens"), kind=price["kind"], where=f"record:{index}")
    _require(bool(entries) and route.data_class in DATA_CLASSES and route.call_kind in CALL_KINDS
             and route.output in OUTPUTS, "model_route_record_invalid")
    return route


def price_row(config: Mapping[str, Any], provider: str, model: str) -> Mapping[str, Any] | None:
    """The provider's published list price for ``model``, or ``None`` when the table has none."""
    return config["prices"].get(provider, {}).get(model)
