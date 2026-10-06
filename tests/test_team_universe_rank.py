"""Hermetic team ranking: the strict rank config, private per-family weights and the tier rule. Synthetic teams only."""
import copy
import hashlib
import json

import pytest

from tests.team_universe_fixture import (
    FIXTURE_STRINGS,
    company,
    one,
    pipeline,
    screen_answers,
    weights,
)
from tools.daily_research import site_screen as ss
from tools.team_universe import rank as tr
from tools.team_universe import universe as tu


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    """pytest's tmp_path is on storage the out-dir guard refuses, a shell may set the worker flag, and Git is slow."""
    monkeypatch.setattr(ss, "VOLATILE_ROOTS", ())
    monkeypatch.delenv("BLUEPRINT_DAILY_RESEARCH_WORKER_ENABLED", raising=False)
    monkeypatch.setattr(ss, "code_state", lambda root=None: {"commit": "0" * 40, "dirty": False, "source": "git"})


@pytest.fixture
def record(tmp_path):
    """A fully evidenced team screen record: a seed-stage palletizing pilot, open, with a learned policy."""
    return one(tmp_path, screen_answers(1))


def config(**changes):
    value = json.loads(tr.CONFIG_PATH.read_text())
    value.update(changes)
    return json.dumps(value).encode()


def assess(record, *, weights_raw=None, **changes):
    return tr.assess(record, tr.load_config(config(**changes) if changes else None), tr.load_weights(weights_raw or weights()))


# --- the strict config --------------------------------------------------------------------------
def test_the_reviewed_rank_config_is_strict_and_scores_out_of_100():
    loaded = tr.load_config()
    assert loaded["schema_version"] == tr.RANK == "blueprint.team-rank.v1"
    assert loaded["sha256"] == hashlib.sha256(tr.CONFIG_PATH.read_bytes()).hexdigest()
    assert loaded["points"] == {"fit": 40, "early_stage": 20, "openness": 20, "policy": 10, "recent_activity": 10}
    assert loaded["thresholds"] == {"beta_candidate": 60, "prospect": 30}
    assert loaded["early_stages"] == ["pre_seed", "seed", "series_a", "series_b", "stealth"]


POINTS = {"fit": 40, "early_stage": 20, "openness": 20, "policy": 10, "recent_activity": 10}


@pytest.mark.parametrize("changes", [
    {"schema_version": "blueprint.team-rank.v0"}, {"extra": 1},
    {"points": dict(POINTS, fit=50)}, {"points": dict(POINTS, fit=40.0)}, {"points": dict(POINTS, fit=True)},
    {"points": {name: value for name, value in POINTS.items() if name != "policy"}},
    {"points": dict(POINTS, fit=-10, early_stage=70)},
    {"early_stages": []}, {"early_stages": ["seed", "seed"]}, {"early_stages": ["series_z"]},
    {"early_stages": ["unknown"]}, {"openness_signals": ["telepathy"]}, {"openness_full_at": 0},
    {"openness_full_at": 6}, {"policy_signals": []}, {"evaluable_signals": [1]}, {"recent_days": 0},
    {"recent_days": 2000}, {"recent_days": "365"}, {"thresholds": {"beta_candidate": 30, "prospect": 60}},
    {"thresholds": {"beta_candidate": 101, "prospect": 30}}, {"thresholds": {"beta_candidate": 60}},
    {"thresholds": {"beta_candidate": 60, "prospect": 0}},
])
def test_a_malformed_rank_config_is_refused(changes):
    with pytest.raises(ss.ScreenError, match="^team_universe_rank_config_invalid$"):
        tr.load_config(config(**changes))


@pytest.mark.parametrize("raw", [b"not json", b"[]", b"{}", b"null"])
def test_a_rank_config_that_is_not_an_object_is_refused(raw):
    with pytest.raises(ss.ScreenError, match="^team_universe_rank_config_invalid$"):
        tr.load_config(raw)


# --- the private family weights -----------------------------------------------------------------
def weights_document(**changes):
    value = {"schema_version": tr.WEIGHTS, "reference": "synthetic-weights-20261005",
             "weights": {"palletizing_depalletizing": 6}}
    value.update(changes)
    return json.dumps(value).encode()


@pytest.mark.parametrize("raw, code", [
    (weights_document(schema_version="blueprint.team-family-weights.v0"), "team_universe_family_weights_invalid"),
    (weights_document(weights={"juggling": 1}), "team_universe_family_weights_invalid"),
    (weights_document(weights={"palletizing_depalletizing": -1}), "team_universe_family_weights_invalid"),
    (weights_document(weights={"palletizing_depalletizing": True}), "team_universe_family_weights_invalid"),
    (weights_document(weights={"palletizing_depalletizing": "6"}), "team_universe_family_weights_invalid"),
    (weights_document(weights={"palletizing_depalletizing": 10 ** 7}), "team_universe_family_weights_invalid"),
    (weights_document(weights={}), "team_universe_family_weights_invalid"),
    (weights_document(extra=1), "team_universe_family_weights_invalid"),
    (weights_document(reference=""), "team_universe_family_weights_invalid"),
    (weights_document(reference=None), "team_universe_family_weights_invalid"),
    (weights_document().replace(b"6}", b"NaN}"), "team_universe_family_weights_invalid"),
    (b"not json", "team_universe_family_weights_invalid"),
    (weights_document(weights={"palletizing_depalletizing": 0, "bimanual_folding": 0}),
     "team_universe_family_weights_empty"),
])
def test_family_weights_are_strict(raw, code):
    with pytest.raises(ss.ScreenError, match=f"^{code}$"):
        tr.load_weights(raw)


def test_weights_scale_to_the_largest_and_a_missing_family_weighs_zero():
    loaded = tr.load_weights(weights(palletizing_depalletizing=6, sorting_pick_and_place=3))
    assert (loaded["source"], loaded["reference"]) == ("weights_file", "synthetic-weights-20261005")
    assert set(loaded["weights"]) == set(loaded["normalized"]) == set(tu.TASK_FAMILIES)
    assert loaded["normalized"]["palletizing_depalletizing"] == 1.0
    assert loaded["normalized"]["sorting_pick_and_place"] == 0.5 and loaded["normalized"]["bimanual_folding"] == 0


def test_a_site_screen_summary_gives_the_weights_by_its_outreach_ready_sites():
    summary = {"schema_version": ss.SUMMARY, "screen": {"tiers_by_focus": {
        "palletizing_depalletizing": {"outreach_ready": 8, "screened": 40},
        "bimanual_folding": {"outreach_ready": 2, "screened": 10},
        "none": {"outreach_ready": 5, "screened": 50}}}}
    loaded = tr.load_weights(json.dumps(summary).encode())
    assert (loaded["source"], loaded["reference"]) == ("site_screen_summary", None)
    assert loaded["weights"]["palletizing_depalletizing"] == 8 and loaded["weights"]["kitting_assembly"] == 0
    assert loaded["normalized"]["bimanual_folding"] == 0.25
    broken = {"schema_version": ss.SUMMARY, "screen": {"tiers_by_focus": {"bimanual_folding": {"outreach_ready": -1}}}}
    with pytest.raises(ss.ScreenError, match="^team_universe_family_weights_invalid$"):
        tr.load_weights(json.dumps(broken).encode())
    unfocused = {"schema_version": ss.SUMMARY, "screen": {"tiers_by_focus": {"none": {"outreach_ready": 3}}}}
    with pytest.raises(ss.ScreenError, match="^team_universe_family_weights_empty$"):
        tr.load_weights(json.dumps(unfocused).encode())


# --- the rule -----------------------------------------------------------------------------------
def test_a_fully_evidenced_early_stage_open_team_with_a_policy_is_a_beta_candidate(record):
    outcome = assess(record)
    assert (outcome["tier"], outcome["score"], outcome["blockers"]) == ("beta_candidate", 100.0, [])
    assert outcome["components"] == {"fit": 40.0, "early_stage": 20, "openness": 20.0, "policy": 10,
                                     "recent_activity": 10}


def test_fit_follows_the_private_weights_and_is_never_hard_coded(record):
    half = assess(record, weights_raw=weights(palletizing_depalletizing=3, sorting_pick_and_place=6))
    assert (half["components"]["fit"], half["score"], half["tier"]) == (20.0, 80.0, "beta_candidate")
    zero = assess(record, weights_raw=weights(sorting_pick_and_place=6))
    assert (zero["components"]["fit"], zero["tier"]) == (0, "prospect")
    assert zero["blockers"] == ["task_family_weight_zero"]


def proven(record, *names, value=False):
    for name in names:
        record["signals"][name]["proven"] = value


@pytest.mark.parametrize("change, tier, blockers", [
    (lambda r: r["contact"].update(route="none"), "prospect", ["contact_route_missing"]),
    (lambda r: proven(r, "learned_policy", "shares_policy", "simulation", "api_sdk"), "prospect", ["not_evaluable"]),
    (lambda r: r["identity"].update(state="unresolved"), "insufficient", ["identity_not_verified"]),
    (lambda r: r["identity"].update(state="contradicted"), "insufficient", ["identity_contradicted"]),
    (lambda r: r["task"].update(proven=[]), "prospect", ["task_family_unproven"]),
    # Still evaluable through simulation, but without a stage, a policy, recent activity or other openness.
    (lambda r: (r["stage"].update(verified=False), proven(r, "learned_policy", "design_partners", "api_sdk",
                                                          "seeking_partners"), r.update(latest_activity=None)),
     "prospect", ["score_below_beta_candidate"]),
    (lambda r: (r["task"].update(proven=[]), r["robot_forms"].update(proven=[])), "insufficient",
     ["task_family_unproven"]),
])
def test_each_beta_requirement_blocks_on_its_own(record, change, tier, blockers):
    changed = copy.deepcopy(record)
    change(changed)
    outcome = assess(changed)
    assert outcome["tier"] == tier and outcome["blockers"] == blockers, outcome


def test_a_low_score_is_insufficient_even_with_a_verified_identity(record):
    bare = copy.deepcopy(record)
    bare["stage"]["verified"] = False
    proven(bare, *tu.SIGNALS)
    bare["latest_activity"] = None
    outcome = assess(bare, weights_raw=weights(sorting_pick_and_place=6))
    assert (outcome["tier"], outcome["score"]) == ("insufficient", 0)


def test_recent_activity_counts_only_inside_the_window(record):
    old = copy.deepcopy(record)
    old["latest_activity"] = "2025-01-01"
    assert assess(old)["components"]["recent_activity"] == 0
    assert assess(old, recent_days=1000)["components"]["recent_activity"] == 10
    future = copy.deepcopy(record)
    future["latest_activity"] = "2027-01-01"
    assert assess(future)["components"]["recent_activity"] == 0


def test_an_unscreened_team_or_a_defect_is_insufficient(record):
    assert assess(None) == {"tier": "insufficient", "score": 0, "components": None, "blockers": ["not_screened"]}
    broken = copy.deepcopy(record)
    del broken["signals"]
    assert assess(broken) == {"tier": "insufficient", "score": 0, "components": None,
                              "blockers": ["rank_computation_unavailable"]}


# --- the rank command ---------------------------------------------------------------------------
def test_rank_writes_a_private_ranked_file_of_prospects_and_returns_counts_only(tmp_path):
    answers = {1: screen_answers(1), 2: screen_answers(2, contact_email="", contact_url=""),
               3: screen_answers(3, company_url="https://directory.example/synthbot-3")}
    # Team 4's discovery quote names no company, so it is held back from the screen.
    discovery = [company(1), company(2), company(3),
                 company(4, source_quote="A palletizing startup raised a seed round to build its robots.")]
    workspace, *_ = pipeline(tmp_path, answers, discovery=discovery)
    result = tr.rank(workspace, weights())
    assert result["tiers"] == {"beta_candidate": 1, "prospect": 1, "insufficient": 2}
    assert (result["teams"], result["screened"], result["weights_source"]) == (4, 3, "weights_file")
    assert result["blockers"] == {"contact_route_missing": 1, "identity_not_verified": 1, "not_screened": 1}
    assert not any(value in ss.canonical(result) for value in FIXTURE_STRINGS)
    path = workspace.root / tu.RANKED_NAME
    assert path.stat().st_mode & 0o777 == 0o600
    ranked = json.loads(path.read_text())
    assert ranked["schema_version"] == tr.RANKED and ranked["config_sha256"] == tr.load_config()["sha256"]
    assert ranked["claim_boundary"] == {"teams_are_prospects": True, "relationship_implied": False,
                                        "contact_is_published_business_route_only": True, "nothing_sent": True}
    rows = ranked["teams"]
    assert [row["domain"] for row in rows] == [f"synthbot-{number}.example" for number in (1, 2, 3, 4)]
    assert [row["tier"] for row in rows] == ["beta_candidate", "prospect", "insufficient", "insufficient"]
    assert all(row["status"] == "prospect" for row in rows)
    assert rows[0]["contact"] == {"route": "role_inbox", "address": "partnerships@synthbot-1.example",
                                  "url": "https://synthbot-1.example/contact"}
    assert rows[0]["task_families"] == ["palletizing_depalletizing"] and rows[3]["screen"] is None
    assert rows[3]["blockers"] == ["not_screened"] and rows[3]["discovery"]["proven"] is False


def test_rank_refuses_bad_weights_or_config_before_writing(tmp_path):
    workspace, *_ = pipeline(tmp_path, {1: screen_answers(1)})
    with pytest.raises(ss.ScreenError, match="^team_universe_family_weights_invalid$"):
        tr.rank(workspace, b"{}")
    with pytest.raises(ss.ScreenError, match="^team_universe_rank_config_invalid$"):
        tr.rank(workspace, weights(), config_raw=b"{}")
    assert not (workspace.root / tu.RANKED_NAME).exists()
