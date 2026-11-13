from __future__ import annotations

import ast
import copy
import dataclasses
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


def _source_settings(module, name: str) -> set:
    """Every value ``module``'s own source passes as ``name`` (a keyword or a dict key), constants resolved."""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    values = set()

    def value(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and hasattr(module, node.id):
            return getattr(module, node.id)
        return ("unresolved", ast.dump(node))

    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == name:
            values.add(value(node.value))
        elif isinstance(node, ast.Dict):
            for key, item in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == name:
                    values.add(value(item))
    return values


def _one_or_none(values: set):
    """The single setting a call site sends, or ``None`` when it sends none."""
    assert len(values) <= 1, values
    return next(iter(values), None)


def _provider(model: str) -> str:
    return {"gpt": "openai", "gemini": "google"}[model.split("-", 1)[0]]


def _today_entries() -> dict[str, tuple[str, str | None, str | None]]:
    """What every role sends today, read from the call sites' own source, never restated here."""
    import dataclasses

    from blueprint_pipeline import (
        atlas_removal_visibility as visibility,
        atlas_scene_planner as planner,
        atlas_whole_view_review as whole_view,
        clean_plate_removal_analysis_gemini as gemini,
        public_scene_sam31_ai_visual_reviewer as sam31,
        task_evaluation_robot_placement_agent as placement,
        task_evaluation_task_parameter_proposal as proposal,
        website_assembly_coverage as coverage,
        website_candidate_ranking as ranking,
        website_image_completion as completion,
        website_image_repair_agent as repair,
        website_object_spec_research as spec,
        website_task_grounding as grounding,
    )
    from blueprint_pipeline import task_object_authoring_backend as backend
    from blueprint_pipeline.task_evaluation_supervisor import agents_sdk

    artisan = backend.ARTISAN_MODELS
    # Only the text roles carry a reasoning setting; compaction and the image calls send none.
    assert {key for key in artisan if "thinking" in key} == {"thinking"}
    registry = json.loads(completion.REGISTRY_PATH.read_text(encoding="utf-8"))
    rows = registry["backends"] if isinstance(registry.get("backends"), list) else list(registry["backends"].values())
    backend_row = next(row for row in rows if row.get("backend_id") == completion.BACKEND_ID)
    review_thinking = _one_or_none(_source_settings(completion, "thinking_level")
                                   | _source_settings(whole_view, "thinking_level"))
    spec_default = {f.name: f.default for f in dataclasses.fields(agents_sdk.AgentsSDKAgentSpec)}["reasoning_effort"]

    def gemini_role(module, model):
        return (_provider(model), model, _one_or_none(_source_settings(module, "thinking_level")))

    def openai_role(module, model):
        return (_provider(model), model, _one_or_none(_source_settings(module, "reasoning_effort")))

    return {
        "artisan_text": (artisan["provider"], artisan["model"], artisan["thinking"]),
        "artisan_compaction": (artisan["provider"], artisan["compaction_model"], None),
        "artisan_image_envelope": (artisan["provider"], artisan["image_model"], None),
        "artisan_image_tool": (artisan["provider"], artisan["image_tool_model"], None),
        "website_image_edit": ("openai", backend_row["execution"]["model_snapshot"], None),
        "website_view_review": (_provider(gemini.DEFAULT_MODEL), gemini.DEFAULT_MODEL, review_thinking),
        "website_task_grounding": gemini_role(grounding, gemini.DEFAULT_MODEL),
        "website_removal_visibility": gemini_role(visibility, gemini.DEFAULT_MODEL),
        "website_assembly_coverage": gemini_role(coverage, coverage.CLASSIFIER_MODEL),
        "website_candidate_ranking": gemini_role(ranking, ranking.MODEL),
        "website_removal_analysis": gemini_role(gemini, gemini.DEFAULT_MODEL),
        "website_movable_obstacles": gemini_role(gemini, gemini.DEFAULT_MODEL),
        "website_spec_research": openai_role(spec, spec.MODEL),
        "task_parameter_proposal": openai_role(proposal, proposal.MODEL),
        "website_image_repair": openai_role(repair, repair.MODEL),
        "robot_placement": openai_role(placement, placement.ROBOT_PLACEMENT_AGENT_MODEL),
        "agents_default": ("openai", agents_sdk.DEFAULT_SUPERVISOR_AGENT_MODEL, spec_default),
        # The website's Atlas runtime policy names the model; the routing entry leaves it to the job.
        "atlas_scene_planner": ("openai", None, _one_or_none(_source_settings(planner, "reasoning_effort"))),
        "sam31_visual_review": openai_role(sam31, sam31.AI_REVIEW_MODEL),
    }


def test_call_site_readers_see_the_settings_they_name() -> None:
    from blueprint_pipeline import (
        clean_plate_removal_analysis_gemini as gemini,
        task_evaluation_robot_placement_agent as placement,
        website_assembly_coverage as coverage,
    )

    assert _source_settings(gemini, "thinking_level") == {"low"}
    assert _source_settings(coverage, "thinking_level") == {coverage.CLASSIFIER_THINKING_LEVEL}
    assert _source_settings(placement, "reasoning_effort") == {placement.ROBOT_PLACEMENT_AGENT_REASONING_EFFORT}


def test_host_overridable_gemini_models_say_so(config: dict) -> None:
    from blueprint_pipeline import clean_plate_removal_analysis_gemini as gemini

    overridable = {role for role, row in config["roles"].items() if "model_override_env" in row}
    assert overridable == {"website_removal_analysis", "website_movable_obstacles"}
    for role in overridable:
        assert config["roles"][role]["model_override_env"] == gemini.MODEL_ENV


def test_shipped_config_is_valid_and_current_is_active(config: dict) -> None:
    assert config["active_profile"] == routing.CURRENT_PROFILE
    assert set(config["profiles"]) == {"current", "same_level_claude"}


def test_every_role_is_covered_by_the_today_table(config: dict) -> None:
    assert set(config["roles"]) == set(_today_entries())


@pytest.mark.parametrize("role", sorted(_today_entries()))
def test_no_configuration_resolves_each_role_to_exactly_today(config: dict, role: str) -> None:
    provider, model, effort = _today_entries()[role]
    assert config["profiles"]["current"][role] == [{"provider": provider, "model": model, "effort": effort}]
    route = resolve_route(role, config=config, job_model="gpt-6.1-sol" if model is None else None)
    assert [entry.as_dict() for entry in route.entries] == [
        {"provider": provider, "model": model or "gpt-6.1-sol", "effort": effort}]
    assert route.skipped == ()
    assert route.profile == "current"
    # The job's Privacy Policy version does not change today's route.
    assert resolve_route(role, config=config, privacy_policy_version="2026-11-12.2",
                         job_model="gpt-6.1-sol" if model is None else None).entries == route.entries


def test_resolving_with_the_shipped_file_reads_the_same_routes(config: dict) -> None:
    for role in config["roles"]:
        assert (resolve_route(role, job_model=_job_model(config, role)).entries
                == resolve_route(role, config=config, job_model=_job_model(config, role)).entries)


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
                   "website_image_repair", "robot_placement", "agents_default", "atlas_scene_planner",
                   "sam31_visual_review"}
SMALL_MODEL_ROLES = {"website_view_review", "website_task_grounding", "website_removal_visibility",
                     "website_assembly_coverage", "website_candidate_ranking"}
NO_CLAUDE_ROLES = {"artisan_image_envelope", "artisan_image_tool", "website_image_edit", "website_removal_analysis",
                   "website_movable_obstacles"}
CONSENT = routing.SiteConsent("2026-09-24.v1", "anthropic", "claude-opus-5-5")
CONSENTED_ROLES = {"artisan_text", "artisan_compaction"}
JOB_MODEL = "gpt-6.1-sol"


def _job_model(config: dict, role: str) -> str | None:
    return JOB_MODEL if config["roles"][role].get("model_source") == "job_policy" else None


def _resolve(role: str, config: dict, **kwargs):
    return resolve_route(role, config=config, job_model=_job_model(config, role), **kwargs)


@pytest.fixture()
def shipped(monkeypatch):
    """Replace the shipped file's content for one test: the one seam the privacy gate has."""
    def install(config: dict) -> dict:
        frozen = copy.deepcopy(config)
        monkeypatch.setattr(routing, "_load_shipped", lambda: copy.deepcopy(frozen))
        return copy.deepcopy(frozen)
    return install


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
def test_privacy_gate_keeps_site_data_away_from_anthropic_without_consent(config: dict, role: str) -> None:
    route = _resolve(role, config, profile="same_level_claude")
    assert all(entry.provider != "anthropic" for entry in route.entries)
    assert [(entry.provider, reason) for entry, reason in route.skipped] == [
        ("anthropic", f"provider_privacy_not_permitted:anthropic:{route.data_class}")]
    assert route.entries == _resolve(role, config).entries


@pytest.mark.parametrize("profile", ["current", "same_level_claude"])
@pytest.mark.parametrize("role", sorted(TOP_MODEL_ROLES | SMALL_MODEL_ROLES | NO_CLAUDE_ROLES))
def test_the_3d_authoring_consent_admits_anthropic_for_its_two_roles_only(config: dict, role: str,
                                                                          profile: str) -> None:
    route = _resolve(role, config, profile=profile, site_consents=(CONSENT,))
    if role in CONSENTED_ROLES and profile == "same_level_claude":
        assert route.default == routing.RouteEntry("anthropic", "claude-opus-5-5", None if role ==
                                                   "artisan_compaction" else "max")
        assert route.skipped == ()
    else:
        assert all(entry.provider != "anthropic" for entry in route.entries)


def test_a_consent_for_another_statement_or_model_admits_nothing(config: dict) -> None:
    config = copy.deepcopy(config)
    config["profiles"]["same_level_claude"]["artisan_text"].insert(
        1, {"provider": "anthropic", "model": "claude-haiku-5-5", "effort": None})
    for consent in (routing.SiteConsent("2026-09-24.v2", "anthropic", "claude-opus-5-5"),
                    routing.SiteConsent("2026-09-24.v1", "anthropic", "claude-haiku-5-5"),
                    routing.SiteConsent("2026-09-24.v1", "openai", "claude-opus-5-5")):
        route = _resolve("artisan_text", config, profile="same_level_claude", site_consents=(consent,))
        assert [entry.provider for entry in route.entries] == ["openai"]
    # The real consent admits its own model, never another Claude model.
    route = _resolve("artisan_text", config, profile="same_level_claude", site_consents=(CONSENT,))
    assert [(entry.provider, entry.model) for entry in route.entries] == [
        ("anthropic", "claude-opus-5-5"), ("openai", "gpt-6.1-sol")]
    assert [entry.model for entry, _ in route.skipped] == ["claude-haiku-5-5"]


def test_consents_must_be_typed(config: dict) -> None:
    with pytest.raises(ModelRoutingError, match="site_consents_invalid"):
        _resolve("artisan_text", config, site_consents=frozenset({"anthropic"}))  # type: ignore[arg-type]
    with pytest.raises(ModelRoutingError, match="site_consent_invalid"):
        routing.SiteConsent("2026-09-24.v1", "zai", "glm")


def test_a_consent_statement_cannot_widen_what_the_privacy_policy_admits(config: dict) -> None:
    bad = copy.deepcopy(config)
    bad["site_consents"]["2026-09-24.v1"]["data_classes"].append("site_video")
    with pytest.raises(ModelRoutingError, match="site_consent_invalid:2026-09-24.v1:site_video"):
        validate_config(bad, today=TODAY)
    bad = copy.deepcopy(config)
    bad["site_consents"]["2026-09-24.v1"]["roles"].append("website_candidate_ranking")
    with pytest.raises(ModelRoutingError, match="site_consent_invalid:2026-09-24.v1:website_candidate_ranking"):
        validate_config(bad, today=TODAY)


def test_privacy_gate_per_data_class_matches_todays_wording(config: dict) -> None:
    permitted = {name: routing.permitted_providers(name) for name in routing.DATA_CLASSES}
    assert permitted == {"site_frames": {"openai", "google"}, "site_video": {"google"},
                         "site_job_text": {"openai", "google"}, "internal_text": {"openai", "anthropic"}}
    by_choice = {name: row["providers_by_site_choice"]
                 for name, row in config["privacy_policies"]["baseline"]["data_classes"].items()}
    assert by_choice == {"site_frames": ["anthropic"], "site_video": [], "site_job_text": [], "internal_text": []}


def test_an_unlisted_privacy_version_gets_the_baseline(config: dict, shipped) -> None:
    config = copy.deepcopy(config)
    wider = copy.deepcopy(config["privacy_policies"]["baseline"])
    wider["data_classes"]["site_frames"]["providers"].append("anthropic")
    wider["data_classes"]["site_frames"]["providers_by_site_choice"] = []
    config["privacy_policies"]["2099-01-01.1"] = wider
    config = shipped(config)
    validate_config(config, today=TODAY)
    new = resolve_route("artisan_text", config=config, profile="same_level_claude",
                        privacy_policy_version="2099-01-01.1")
    assert new.default.provider == "anthropic" and new.privacy_policy == "2099-01-01.1"
    assert new.record()["privacy_policy_version"] == "2099-01-01.1"
    for version in (None, "2026-11-12.2", "2026-11-12.1", "baseline", "", " 2099-01-01.1"):
        old = resolve_route("artisan_text", config=config, profile="same_level_claude",
                            privacy_policy_version=version)
        assert old.default.provider == "openai" and old.privacy_policy == "baseline"
        assert old.privacy_policy_version == version and old.record()["privacy_policy_version"] == version


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


def test_a_job_policy_model_must_be_named_and_priced(config: dict) -> None:
    with pytest.raises(ModelRoutingError, match="job_model_required:atlas_scene_planner"):
        resolve_route("atlas_scene_planner", config=config)
    with pytest.raises(ModelRoutingError, match="unpriced:current:atlas_scene_planner:openai:gpt-5.6-sol"):
        resolve_route("atlas_scene_planner", config=config, job_model="gpt-5.6-sol")
    bad = copy.deepcopy(config)
    bad["profiles"]["current"]["website_spec_research"][0]["model"] = None
    with pytest.raises(ModelRoutingError, match="model_routing_model_invalid:current:website_spec_research"):
        validate_config(bad, today=TODAY)


def test_a_job_model_for_a_role_that_takes_none_is_refused(config: dict) -> None:
    with pytest.raises(ModelRoutingError, match="model_routing_job_model_not_taken:robot_placement"):
        resolve_route("robot_placement", config=config, job_model="claude-opus-5-5")


def test_every_model_the_planner_admits_today_resolves_at_the_repositorys_price(config: dict) -> None:
    from blueprint_pipeline import openai_prompt_cache

    def planner_admits(model: str) -> bool:  # atlas_scene_planner.py:697
        return model.startswith("gpt-6") and openai_prompt_cache.pricing_for_model(model) is not None

    families = sorted({pricing.model_family for pricing in openai_prompt_cache._GPT56_PRICING.values()})
    candidates = [model for family in families for model in (family, family + "-2026-09-01")]
    admitted = [model for model in candidates if planner_admits(model)]
    assert {model for model in admitted if "-2026-" not in model} == {
        "gpt-6-astra", "gpt-6.1-sol", "gpt-6-sol", "gpt-6-luna"}
    for model in admitted:
        route = resolve_route("atlas_scene_planner", config=config, job_model=model)
        assert route.default.model == model
        pricing = openai_prompt_cache.pricing_for_model(model)
        listed = route.list_price(0)
        assert dict(listed["usd_per_million_tokens"]) == {
            "input": pricing.uncached_input_per_million_usd, "cache_write": pricing.cache_write_per_million_usd,
            "cache_read": pricing.cached_read_per_million_usd, "output": pricing.output_per_million_usd}
        long = listed["long_prompt"]
        assert long["above_input_tokens"] == pricing.long_context_threshold_tokens
        assert long["usd_per_million_tokens"]["input"] == pytest.approx(
            pricing.uncached_input_per_million_usd * pricing.long_context_input_multiplier)
        assert long["usd_per_million_tokens"]["output"] == pytest.approx(
            pricing.output_per_million_usd * pricing.long_context_output_multiplier)
    for model in (model for model in candidates if not planner_admits(model)):
        with pytest.raises(ModelRoutingError, match="unpriced"):
            resolve_route("atlas_scene_planner", config=config, job_model=model)


# --- the gate reads only the shipped privacy tables (the reviewer's probes, inverted) ---


def _loosened(config: dict) -> list[tuple[str, dict]]:
    widened = copy.deepcopy(config)
    widened["site_consents"] = {}
    for name in ("site_frames", "site_job_text", "site_video"):
        row = widened["privacy_policies"]["baseline"]["data_classes"][name]
        row["providers"] = sorted(set(row["providers"]) | {"anthropic"})
        row["providers_by_site_choice"] = []
    rewritten = copy.deepcopy(config)
    rewritten["site_consents"]["2026-09-24.v1"]["model"] = "claude-haiku-5-5"
    rewritten["site_consents"]["2026-09-24.v1"]["roles"] = [
        role for role, row in config["roles"].items() if row["data_class"] == "site_frames"]
    versioned = copy.deepcopy(config)
    wide = copy.deepcopy(config["privacy_policies"]["baseline"])
    wide["data_classes"]["site_frames"]["providers"] = ["openai", "google", "anthropic"]
    wide["data_classes"]["site_frames"]["providers_by_site_choice"] = []
    versioned["privacy_policies"]["2026-11-12.2"] = wide
    relabelled = copy.deepcopy(config)
    relabelled["roles"]["website_candidate_ranking"]["data_class"] = "site_frames"
    return [("privacy_policies", widened), ("site_consents", rewritten), ("privacy_policies", versioned),
            ("role_data_classes", relabelled)]


@pytest.mark.parametrize("index", range(4))
def test_a_loosened_configuration_is_refused_everywhere(config: dict, index: int, tmp_path) -> None:
    part, loosened = _loosened(config)[index]
    code = f"model_routing_privacy_not_shipped:{part}"
    with pytest.raises(ModelRoutingError, match=code):
        validate_config(loosened, today=TODAY)
    path = tmp_path / "model_routing.v1.json"
    path.write_text(json.dumps(loosened), encoding="utf-8")
    with pytest.raises(ModelRoutingError, match=code):
        routing.load_config(path, today=TODAY)
    for role in ("website_view_review", "website_candidate_ranking", "website_removal_analysis", "robot_placement"):
        with pytest.raises(ModelRoutingError, match=code):
            resolve_route(role, config=loosened, profile="same_level_claude", privacy_policy_version="2026-11-12.2",
                          site_consents=(routing.SiteConsent("2026-09-24.v1", "anthropic", "claude-haiku-5-5"),))
    record = resolve_route("robot_placement", config=config).record()
    with pytest.raises(ModelRoutingError, match=code):
        route_from_record(record, config=loosened)


def test_the_gate_takes_no_table_from_its_caller() -> None:
    import inspect

    assert "config" not in inspect.signature(routing.privacy_refusal).parameters
    assert "config" not in inspect.signature(routing.permitted_providers).parameters
    assert "config" not in inspect.signature(routing.privacy_policy_key).parameters
    # Video stays with Google, whatever a caller would like.
    assert routing.privacy_refusal(role="website_removal_analysis",
                                   entry=routing.RouteEntry("anthropic", "claude-opus-5-5", None),
                                   privacy_policy_version=None, site_consents=(CONSENT,)) == (
        "provider_privacy_not_permitted:anthropic:site_video")


def test_no_route_can_hold_an_entry_the_gate_refuses(config: dict) -> None:
    route = resolve_route("robot_placement", config=config)
    opus = routing.RouteEntry("anthropic", "claude-opus-5-5", "high")
    opus_price = config["prices"]["anthropic"]["claude-opus-5-5"]
    with pytest.raises(ModelRoutingError, match="model_route_entry_provider_privacy_not_permitted:anthropic"):
        dataclasses.replace(route, entries=(opus,), list_prices=(opus_price,))  # a forbidden default
    with pytest.raises(ModelRoutingError, match="model_route_entry_provider_privacy_not_permitted:anthropic"):
        dataclasses.replace(route, entries=route.entries + (opus,), list_prices=route.list_prices + (opus_price,))
    with pytest.raises(ModelRoutingError, match="model_route_role_class_mismatch"):
        dataclasses.replace(route, data_class="internal_text")
    # The residual this module states: the same data class under another role's name is not detectable here.
    relabelled = dataclasses.replace(route, role="artisan_text", site_consents=(CONSENT,), entries=(
        routing.RouteEntry("anthropic", "claude-opus-5-5", None), *route.entries),
        list_prices=(opus_price, *route.list_prices))
    assert relabelled.default.provider == "anthropic"


def test_a_restored_route_says_whether_its_configuration_is_current(config: dict) -> None:
    record = resolve_route("robot_placement", config=config).record()
    assert route_from_record(record, config=config).config_digest_matches_current is True
    changed = copy.deepcopy(config)
    changed["prices"]["openai"]["gpt-6.1-sol"]["usd_per_million_tokens"]["output"] = 11.0
    restored = route_from_record(record, config=changed)
    assert restored.config_digest_matches_current is False
    assert restored.list_price(0)["usd_per_million_tokens"]["output"] == 10.0


def test_the_shipped_job_text_row_carries_its_note(config: dict) -> None:
    note = config["privacy_policies"]["baseline"]["data_classes"]["site_job_text"]["note"]
    assert "what runs today" in note and "new Privacy version" in note
    for role in ("website_removal_analysis", "website_movable_obstacles"):
        assert "documentation only" in config["roles"][role]["today"]


# --- freezing ----------------------------------------------------------------


def test_a_resolved_route_is_frozen_with_its_list_prices(config: dict) -> None:
    route = resolve_route("artisan_text", config=config, profile="same_level_claude", site_consents=(CONSENT,))
    record = route.record()
    assert [entry["list_price"]["usd_per_million_tokens"]["output"] for entry in record["entries"]] == [20.0, 10.0]
    assert record["site_consents"] == [CONSENT.as_dict()]
    changed = copy.deepcopy(config)
    changed["profiles"]["same_level_claude"]["artisan_text"].reverse()
    changed["prices"]["anthropic"]["claude-opus-5-5"]["usd_per_million_tokens"]["output"] = 99.0
    restored = route_from_record(json.loads(json.dumps(record)), config=changed)
    assert restored == route
    assert restored.list_price(0)["usd_per_million_tokens"]["output"] == 20.0
    assert resolve_route("artisan_text", config=changed, profile="same_level_claude",
                         site_consents=(CONSENT,)).record()["route_digest"] != record["route_digest"]


def _reseal(body: dict) -> dict:
    body = {key: value for key, value in body.items() if key != "route_digest"}
    return {**body, "route_digest": routing._digest(body)}


def test_a_restored_record_is_gated_again_whatever_its_digest(config: dict) -> None:
    """The reviewer's probe, inverted: a resealed record that adds Claude first is refused."""
    record = resolve_route("artisan_text", config=config).record()
    record["entries"].insert(0, {"provider": "anthropic", "model": "claude-opus-5-5", "effort": "max",
                                 "list_price": config["prices"]["anthropic"]["claude-opus-5-5"]})
    with pytest.raises(ModelRoutingError, match="model_route_record_provider_privacy_not_permitted:anthropic"):
        route_from_record(_reseal(record), config=config)
    # A consent written into the record admits only what the consent statement covers.
    record["site_consents"] = [CONSENT.as_dict()]
    assert [entry.provider for entry in route_from_record(_reseal(record), config=config).entries] == [
        "anthropic", "openai"]
    forged = resolve_route("website_spec_research", config=config).record()
    forged["entries"].insert(0, {"provider": "anthropic", "model": "claude-opus-5-5", "effort": None,
                                 "list_price": config["prices"]["anthropic"]["claude-opus-5-5"]})
    forged["site_consents"] = [CONSENT.as_dict()]
    with pytest.raises(ModelRoutingError, match="privacy_not_permitted:anthropic:site_frames"):
        route_from_record(_reseal(forged), config=config)


@pytest.mark.parametrize("field, value, code", [
    ("data_class", "internal_text", "model_route_record_role_mismatch"),
    ("role", "nope", "model_route_record_role_mismatch"),
    ("privacy_policy", "2099-01-01.1", "model_route_record_privacy_key_mismatch"),
    ("site_consents", [{"statement_version": "v", "provider": "zai", "model": "m"}], "site_consent_invalid"),
])
def test_a_restored_record_must_match_the_configuration(config: dict, field: str, value, code: str) -> None:
    record = resolve_route("artisan_text", config=config).record()
    record[field] = value
    with pytest.raises(ModelRoutingError, match=code):
        route_from_record(_reseal(record), config=config)


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


def test_a_resolved_route_does_not_move_when_the_configuration_changes(config: dict) -> None:
    route = resolve_route("artisan_text", config=config, profile="same_level_claude", site_consents=(CONSENT,))
    before = route.record()
    config["prices"]["anthropic"]["claude-opus-5-5"]["usd_per_million_tokens"]["output"] = 99.0
    config["prices"]["anthropic"]["claude-opus-5-5"]["source"] = "https://example.invalid/"
    config["profiles"]["same_level_claude"]["artisan_text"][0]["effort"] = "low"
    assert route.list_price(0)["usd_per_million_tokens"]["output"] == 20.0
    assert route.default.effort == "max"
    assert route.record() == before


def test_a_restored_route_does_not_move_when_its_input_record_changes(config: dict) -> None:
    record = resolve_route("artisan_text", config=config, profile="same_level_claude",
                           site_consents=(CONSENT,)).record()
    restored = route_from_record(record, config=config)
    digest = restored.record()["route_digest"]
    record["entries"][0]["list_price"]["usd_per_million_tokens"]["output"] = 99.0
    record["entries"][0]["model"] = "claude-opus-5"
    record["skipped"].append({"provider": "openai", "model": "x", "effort": None, "reason": "r"})
    record["site_consents"].append({"statement_version": "x", "provider": "google", "model": "m"})
    assert restored.list_price(0)["usd_per_million_tokens"]["output"] == 20.0
    assert restored.default.model == "claude-opus-5-5"
    assert restored.site_consents == (CONSENT,) and restored.skipped == ()
    assert restored.record()["route_digest"] == digest


def test_frozen_prices_hand_out_nothing_a_caller_can_change(config: dict) -> None:
    route = resolve_route("artisan_text", config=config)
    price = route.list_price(0)
    with pytest.raises(TypeError):
        price["kind"] = "image"  # type: ignore[index]
    with pytest.raises(TypeError):
        price["usd_per_million_tokens"]["output"] = 0.0  # type: ignore[index]
    copy_ = route.plain_list_price(0)
    copy_["usd_per_million_tokens"]["output"] = 0.0
    record = route.record()
    record["entries"][0]["list_price"]["usd_per_million_tokens"]["output"] = 0.0
    assert route.list_price(0)["usd_per_million_tokens"]["output"] == 10.0
    assert isinstance(route.site_consents, tuple) and isinstance(route.skipped, tuple)


def test_a_role_whose_call_site_discloses_frames_is_classed_with_frames(config: dict) -> None:
    from blueprint_pipeline import task_evaluation_task_parameter_proposal as proposal
    from blueprint_pipeline import website_object_spec_research as spec

    # The proposal's authority field is a private derived-frame disclosure; spec research sends images.
    assert "private_derived_frame_disclosure_authorized" in Path(proposal.__file__).read_text(encoding="utf-8")
    assert _source_settings(spec, "type") >= {"input_image"}
    for role in ("task_parameter_proposal", "website_spec_research"):
        assert config["roles"][role]["data_class"] == "site_frames"
