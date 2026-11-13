from __future__ import annotations

import ast
import copy
import json
import math
from datetime import date
from pathlib import Path

import pytest

from blueprint_pipeline import model_routing as routing
from blueprint_pipeline.model_routing import ModelRoutingError, resolve_route, route_from_record, validate_config

ROOT = Path(__file__).resolve().parents[1]
TODAY = date(2026, 11, 13)


@pytest.fixture()
def config() -> dict:
    return routing.load_config(today=TODAY)


def _today_entries() -> dict[str, tuple[str, str, str | None]]:
    """What every role sends today, read from the call sites' own constants."""
    from blueprint_pipeline import (
        clean_plate_removal_analysis_gemini as gemini,
        task_evaluation_robot_placement_agent as placement,
        task_evaluation_task_parameter_proposal as proposal,
        website_assembly_coverage as coverage,
        website_candidate_ranking as ranking,
        website_image_completion as completion,
        website_image_repair_agent as repair,
        website_object_spec_research as spec,
    )
    from blueprint_pipeline.task_evaluation_supervisor import agents_sdk
    from blueprint_pipeline.task_object_authoring_backend import ARTISAN_MODELS as artisan

    registry = json.loads(completion.REGISTRY_PATH.read_text(encoding="utf-8"))
    rows = registry["backends"] if isinstance(registry.get("backends"), list) else list(registry["backends"].values())
    backend = next(row for row in rows if row.get("backend_id") == completion.BACKEND_ID)
    edit_model = backend["execution"]["model_snapshot"]
    return {
        "artisan_text": (artisan["provider"], artisan["model"], artisan["thinking"]),
        "artisan_compaction": (artisan["provider"], artisan["compaction_model"], None),
        "artisan_image_envelope": (artisan["provider"], artisan["image_model"], None),
        "artisan_image_tool": (artisan["provider"], artisan["image_tool_model"], None),
        "website_image_edit": ("openai", edit_model, None),
        "website_view_review": ("google", gemini.DEFAULT_MODEL, None),
        "website_task_grounding": ("google", gemini.DEFAULT_MODEL, None),
        "website_removal_visibility": ("google", gemini.DEFAULT_MODEL, None),
        "website_assembly_coverage": ("google", coverage.CLASSIFIER_MODEL, coverage.CLASSIFIER_THINKING_LEVEL),
        "website_candidate_ranking": ("google", ranking.MODEL, None),
        "website_removal_analysis": ("google", gemini.DEFAULT_MODEL, None),
        "website_spec_research": ("openai", spec.MODEL, None),
        "task_parameter_proposal": ("openai", proposal.MODEL, "high"),
        "website_image_repair": ("openai", repair.MODEL, None),
        "robot_placement": ("openai", placement.ROBOT_PLACEMENT_AGENT_MODEL,
                            placement.ROBOT_PLACEMENT_AGENT_REASONING_EFFORT),
        "agents_default": ("openai", agents_sdk.DEFAULT_SUPERVISOR_AGENT_MODEL, None),
    }


def test_shipped_config_is_valid_and_current_is_active(config: dict) -> None:
    assert config["active_profile"] == routing.CURRENT_PROFILE
    assert set(config["profiles"]) == {"current", "same_level_claude"}


def test_every_role_is_covered_by_the_today_table(config: dict) -> None:
    assert set(config["roles"]) == set(_today_entries())


@pytest.mark.parametrize("role", sorted(_today_entries()))
def test_no_configuration_resolves_each_role_to_exactly_today(config: dict, role: str) -> None:
    route = resolve_route(role, config=config)
    provider, model, effort = _today_entries()[role]
    assert [entry.as_dict() for entry in route.entries] == [{"provider": provider, "model": model, "effort": effort}]
    assert route.skipped == ()
    assert route.profile == "current"
    # The job's Privacy Policy version does not change today's route.
    assert resolve_route(role, config=config, privacy_policy_version="2026-11-12.2").entries == route.entries


def test_resolving_with_the_shipped_file_reads_the_same_routes(config: dict) -> None:
    for role in config["roles"]:
        assert resolve_route(role).entries == resolve_route(role, config=config).entries


def test_only_the_routing_tests_and_the_failover_module_import_the_routing_modules() -> None:
    importers = set()
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [alias.name for alias in node.names]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            if any(name.split(".")[-1] in {"model_routing", "model_failover"} for name in names):
                importers.add(path.relative_to(ROOT).as_posix())
    assert importers == {"src/blueprint_pipeline/model_failover.py"}


TOP_MODEL_ROLES = {"artisan_text", "artisan_compaction", "website_spec_research", "task_parameter_proposal",
                   "website_image_repair", "robot_placement", "agents_default"}
SMALL_MODEL_ROLES = {"website_view_review", "website_task_grounding", "website_removal_visibility",
                     "website_assembly_coverage", "website_candidate_ranking"}
NO_CLAUDE_ROLES = {"artisan_image_envelope", "artisan_image_tool", "website_image_edit", "website_removal_analysis"}


def test_same_level_mapping_exists_but_is_not_the_default(config: dict) -> None:
    same = config["profiles"]["same_level_claude"]
    current = config["profiles"]["current"]
    assert TOP_MODEL_ROLES | SMALL_MODEL_ROLES | NO_CLAUDE_ROLES == set(config["roles"])
    for role in TOP_MODEL_ROLES:
        assert same[role][0]["model"] == "claude-opus-5-5" and same[role][1:] == current[role]
    for role in SMALL_MODEL_ROLES:
        assert same[role][0]["model"] == "claude-haiku-5-5" and same[role][1:] == current[role]
    for role in NO_CLAUDE_ROLES:
        assert same[role] == current[role]
    # Effort carries across where today sets one; no setting stays no setting.
    for role in TOP_MODEL_ROLES | SMALL_MODEL_ROLES:
        today = current[role][0]["effort"]
        assert same[role][0]["effort"] == (today.lower() if today else None)


# --- the privacy gate --------------------------------------------------------


@pytest.mark.parametrize("role", sorted(TOP_MODEL_ROLES | SMALL_MODEL_ROLES))
def test_privacy_gate_keeps_site_data_away_from_anthropic_without_the_sites_choice(config: dict, role: str) -> None:
    route = resolve_route(role, config=config, profile="same_level_claude")
    assert all(entry.provider != "anthropic" for entry in route.entries)
    assert [(entry.provider, reason) for entry, reason in route.skipped] == [
        ("anthropic", f"provider_privacy_not_permitted:anthropic:{route.data_class}")]
    assert route.entries == resolve_route(role, config=config).entries


@pytest.mark.parametrize("role", sorted(TOP_MODEL_ROLES | SMALL_MODEL_ROLES))
def test_a_sites_explicit_choice_admits_anthropic_for_frames_only(config: dict, role: str) -> None:
    route = resolve_route(role, config=config, profile="same_level_claude", site_choices=frozenset({"anthropic"}))
    if config["roles"][role]["data_class"] == "site_frames":
        assert route.default.provider == "anthropic" and route.skipped == ()
    else:
        assert route.default.provider != "anthropic"


def test_privacy_gate_per_data_class_matches_todays_wording(config: dict) -> None:
    permitted = {name: routing.permitted_providers(config, name) for name in routing.DATA_CLASSES}
    assert permitted == {"site_frames": {"openai", "google"}, "site_video": {"google"},
                         "site_job_text": {"openai", "google"}, "internal_text": {"openai", "anthropic"}}
    chosen = frozenset({"anthropic", "openai", "google"})
    assert routing.permitted_providers(config, "site_video", site_choices=chosen) == {"google"}
    assert routing.permitted_providers(config, "site_job_text", site_choices=chosen) == {"openai", "google"}


def test_an_unlisted_privacy_version_gets_the_baseline(config: dict) -> None:
    config = copy.deepcopy(config)
    wider = copy.deepcopy(config["privacy_policies"]["baseline"])
    wider["data_classes"]["site_frames"]["providers"].append("anthropic")
    wider["data_classes"]["site_frames"]["providers_by_site_choice"] = []
    config["privacy_policies"]["2099-01-01.1"] = wider
    validate_config(config, today=TODAY)
    new = resolve_route("artisan_text", config=config, profile="same_level_claude",
                        privacy_policy_version="2099-01-01.1")
    assert new.default.provider == "anthropic" and new.privacy_policy == "2099-01-01.1"
    for version in (None, "2026-11-12.2", "2026-11-12.1", "baseline", ""):
        old = resolve_route("artisan_text", config=config, profile="same_level_claude",
                            privacy_policy_version=version)
        assert old.default.provider == "openai" and old.privacy_policy == "baseline"


def test_a_privacy_version_may_not_narrow_the_baseline(config: dict) -> None:
    config = copy.deepcopy(config)
    narrower = copy.deepcopy(config["privacy_policies"]["baseline"])
    narrower["data_classes"]["site_frames"]["providers"] = ["openai"]
    config["privacy_policies"]["2099-01-01.1"] = narrower
    with pytest.raises(ModelRoutingError, match="narrows_baseline:2099-01-01.1:site_frames"):
        validate_config(config, today=TODAY)


def test_a_role_without_a_permitted_default_is_refused(config: dict) -> None:
    config = copy.deepcopy(config)
    config["profiles"]["same_level_claude"]["website_view_review"] = [
        {"provider": "anthropic", "model": "claude-haiku-5-5", "effort": None}]
    with pytest.raises(ModelRoutingError, match="no_permitted_default:same_level_claude:website_view_review"):
        validate_config(config, today=TODAY)


def test_an_image_role_never_routes_to_a_text_only_provider(config: dict) -> None:
    config = copy.deepcopy(config)
    config["profiles"]["same_level_claude"]["website_image_edit"].insert(
        0, {"provider": "anthropic", "model": "claude-opus-5-5", "effort": None})
    with pytest.raises(ModelRoutingError, match="image_output_unsupported"):
        validate_config(config, today=TODAY)


# --- freezing ----------------------------------------------------------------


def test_a_resolved_route_is_frozen_with_its_list_prices(config: dict) -> None:
    route = resolve_route("artisan_text", config=config, profile="same_level_claude",
                          site_choices=frozenset({"anthropic"}))
    record = route.record()
    assert [entry["list_price"]["usd_per_million_tokens"]["output"] for entry in record["entries"]] == [20.0, 10.0]
    changed = copy.deepcopy(config)
    changed["profiles"]["same_level_claude"]["artisan_text"].reverse()
    changed["prices"]["anthropic"]["claude-opus-5-5"]["usd_per_million_tokens"]["output"] = 99.0
    restored = route_from_record(json.loads(json.dumps(record)))
    assert restored == route
    assert restored.list_price(0)["usd_per_million_tokens"]["output"] == 20.0
    assert resolve_route("artisan_text", config=changed, profile="same_level_claude",
                         site_choices=frozenset({"anthropic"})).record()["route_digest"] != record["route_digest"]


def test_a_tampered_route_record_is_refused(config: dict) -> None:
    record = resolve_route("artisan_text", config=config).record()
    record["entries"][0]["model"] = "gpt-6-astra"
    with pytest.raises(ModelRoutingError, match="model_route_record_digest_mismatch"):
        route_from_record(record)


def test_frozen_routes_are_immutable(config: dict) -> None:
    route = resolve_route("robot_placement", config=config)
    with pytest.raises(AttributeError):
        route.entries = ()  # type: ignore[misc]
    assert isinstance(route.entries, tuple)


# --- prices ------------------------------------------------------------------


def test_list_prices_agree_with_the_repositorys_own_tables(config: dict) -> None:
    from blueprint_pipeline.openai_prompt_cache import pricing_for_model
    from blueprint_pipeline.website_gemini_receipts import gemini_quote

    sol = pricing_for_model("gpt-6.1-sol")
    assert config["prices"]["openai"]["gpt-6.1-sol"]["usd_per_million_tokens"] == {
        "input": sol.uncached_input_per_million_usd, "cache_write": sol.cache_write_per_million_usd,
        "cache_read": sol.cached_read_per_million_usd, "output": sol.output_per_million_usd}
    flash = config["prices"]["google"]["gemini-3.8-flash"]["usd_per_million_tokens"]
    for tokens_in, tokens_out in ((1_000_000, 1), (1, 65_536), (123_457, 4_321)):
        listed = (tokens_in * flash["input"] + tokens_out * flash["output"]) / 1_000_000
        # The quote rounds up to whole cents.
        assert gemini_quote(model="gemini-3.8-flash", input_tokens=tokens_in,
                            max_output_tokens=tokens_out) == math.ceil(round(listed * 100, 9)) / 100
    from blueprint_pipeline import website_image_completion as completion
    registry = json.loads(completion.REGISTRY_PATH.read_text(encoding="utf-8"))
    rows = registry["backends"] if isinstance(registry.get("backends"), list) else list(registry["backends"].values())
    backend = next(row for row in rows if row.get("backend_id") == completion.BACKEND_ID)
    rates = backend["execution"]["pricing_binding"]["usd_per_million_tokens"]
    for model in ("gpt-image-2.5-sunburst", "gpt-image-2.5-sunburst-2026-09-08"):
        row = config["prices"]["openai"][model]["usd_per_million_tokens"]
        assert row == {"input_text": rates["input_text_tokens"], "input_image": rates["input_image_tokens"],
                       "output_image": rates["output_image_tokens"]}


def test_claude_list_prices_are_the_published_rates(config: dict) -> None:
    opus = config["prices"]["anthropic"]["claude-opus-5-5"]
    assert opus["usd_per_million_tokens"] == {"input": 4.0, "cache_write": 5.0, "cache_write_1h": 8.0,
                                              "cache_read": 0.2, "output": 20.0}
    haiku = config["prices"]["anthropic"]["claude-haiku-5-5"]
    assert haiku["usd_per_million_tokens"]["output"] == 0.5
    assert haiku["long_prompt"] == {"above_input_tokens": 100_000, "usd_per_million_tokens": {
        "input": 0.5, "cache_write": 0.625, "cache_write_1h": 1.0, "cache_read": 0.05, "output": 2.5}}


@pytest.mark.parametrize("mutate, code", [
    (lambda c: c["prices"]["openai"]["gpt-6.1-sol"]["usd_per_million_tokens"].update(output=-1), "price_invalid"),
    (lambda c: c["prices"]["openai"]["gpt-6.1-sol"]["usd_per_million_tokens"].update(output=float("nan")),
     "price_invalid"),
    (lambda c: c["prices"]["openai"]["gpt-6.1-sol"]["usd_per_million_tokens"].pop("output"), "price_fields_invalid"),
    (lambda c: c["prices"]["openai"]["gpt-6.1-sol"].update(read_on="2099-01-01"), "price_date_future"),
    (lambda c: c["prices"]["openai"]["gpt-6.1-sol"].update(source="memory"), "price_source_missing"),
    (lambda c: c["prices"]["anthropic"].pop("claude-haiku-5-5"), "unpriced"),
    (lambda c: c["profiles"]["current"]["artisan_text"][0].update(effort="ultra"), "effort_invalid"),
    (lambda c: c["profiles"]["current"]["artisan_text"].append(dict(c["profiles"]["current"]["artisan_text"][0])),
     "entries_duplicate"),
    (lambda c: c["profiles"]["current"].pop("agents_default"), "profile_roles_mismatch"),
    (lambda c: c.update(active_profile="nope"), "active_profile_unknown"),
    (lambda c: c["profiles"]["current"]["artisan_text"][0].update(provider="zai"), "provider_unknown"),
])
def test_invalid_configurations_are_refused(config: dict, mutate, code: str) -> None:
    config = copy.deepcopy(config)
    mutate(config)
    with pytest.raises(ModelRoutingError, match=code):
        validate_config(config, today=TODAY)


def test_unknown_role_and_profile_are_refused(config: dict) -> None:
    with pytest.raises(ModelRoutingError, match="role_unknown"):
        resolve_route("nope", config=config)
    with pytest.raises(ModelRoutingError, match="profile_unknown"):
        resolve_route("artisan_text", config=config, profile="nope")
