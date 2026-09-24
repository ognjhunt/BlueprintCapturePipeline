"""ADP-030/day 28: published object specs from a stated or read identity, with no live calls."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from blueprint_pipeline import website_object_spec_research as research
from blueprint_pipeline import website_task_context as control
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file

SPEC = "https://www.bosch-home.com/us/specs/SHPM88Z75N"
RETAIL = "https://retailer.example/p"
SPEC_PAGE = (b"<html><head><title>Specs</title></head><body><script>var w='99 lb';</script><table>"
             b"<tr><th>Width</th><td>23 9/16 in (59.8 cm)</td></tr><tr><th>Height</th><td>33 7/8 in</td></tr>"
             b"<tr><th>Depth</th><td>22 1/2 in</td></tr><tr><th>Net weight</th><td>97 lbs</td></tr></table>"
             b"<p>Upper rack max load: 10 kg</p></body></html>")
PAGES = {SPEC: {"status": 200, "headers": {"Content-Type": "text/html; charset=utf-8"}, "body": SPEC_PAGE},
         RETAIL: {"status": 301, "headers": {"Location": "https://www.retailer.example/p2"}, "body": b""},
         "https://www.retailer.example/p2": {"status": 200, "headers": {"Content-Type": "text/html"},
                                             "body": b"<div>Overall height: 34 1/2 in</div>"}}
BODY = {"width_m": 0.6, "height_m": 0.86, "depth_m": 0.55}


def _figure(name, value, unit, url, quote):
    return {"name": name, "value": value, "unit": unit, "source_url": url, "quote": quote}


FIGURES = [
    _figure("overall_width", 59.8, "cm", SPEC, "Width 23 9/16 in (59.8 cm)"),
    _figure("overall_height", 33.875, "in", SPEC, "Height 33 7/8 in"),
    _figure("overall_height", 34.5, "in", RETAIL, "Overall height: 34 1/2 in"),  # Cited before its redirect.
    _figure("overall_depth", 22.5, "in", SPEC, "Depth 22 1/2 in"),
    _figure("net_weight", 79, "lb", SPEC, "Net weight 97 lbs"),
    _figure("door_weight", 9, "kg", "https://unfetched.example/door", "Door weight 9 kg"),
    _figure("upper_rack_max_load", 12, "kg", SPEC, "Upper rack max load: 12 kg"),
    _figure("drawer_max_load", 99, "lb", SPEC, "var w='99 lb'"),  # Script text is not page text.
]


def _findings(figures=FIGURES, *, model="SHPM88Z75N", match="exact_model"):
    return {"product": {"brand": "Bosch", "model": model, "model_family": "800 Series", "match": match},
            "figures": figures, "silent_on": ["cutout_width"]}


def _transport(calls):
    def transport(url, *, max_bytes, timeout):
        calls.append(url)
        return PAGES[url]
    return transport


def _public(host, port, proto=0):
    return [(2, 1, 6, "", ("93.184.216.34", port))]


class _Invoker:
    def __init__(self, findings, fetches=(SPEC, RETAIL)):
        self.findings, self.fetches, self.calls = findings, fetches, []

    def configure_reservation_audit(self, **_kwargs):
        pass

    def invoke(self, spec, input_value):
        self.calls.append((spec, input_value))
        for url in self.fetches:
            spec.tool_bindings[0].invoke({"url": url})
        return SimpleNamespace(output=research.ObjectSpecFindings.model_validate(self.findings), model=research.MODEL,
                               usage={"input_tokens": 9000, "hosted_tool_calls": 2}, cost_usd=0.21)


@pytest.fixture
def website(monkeypatch):
    calls = {"reserve": [], "settle": [], "fetch": []}

    def reserve(**kwargs):
        calls["reserve"].append(kwargs)
        return {"status": "admitted", "allocation_binding_digest": kwargs["binding_digest"]}, object()

    def webapp(**kwargs):
        calls["settle"].append(kwargs["payload"]["settlement"])
        return {**kwargs["payload"]["settlement"], "status": "settled"}
    monkeypatch.setattr(control, "reserve_website_preparation_spend", reserve)
    monkeypatch.setattr(control, "website_webapp_request", webapp)
    monkeypatch.setenv(research.ENABLE_ENV, "1")
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS", "1")
    return calls


def _context(**answers):
    value = {"schema_version": "website_site_task_context.v1", "request_id": "req", "scene_id": "scene",
             "capture_id": "capture", "confirmed": True, "confirmed_at": "2026-09-19T00:00:00Z",
             "description": "Open and close the dishwasher", "operator_answers": answers}
    value["context_digest"] = canonical_digest(value, digest_field="context_digest")
    return value


OWNER = _context(item_make_model="Bosch SHPM88Z75N")


def _coverage(tmp_path, labels=(("decoded-000000001", ["BOSCH", "800 Series"]),), body=BODY):
    rows = []
    for index, (frame_id, text) in enumerate(labels):
        path = tmp_path / f"{frame_id}.png"
        Image.new("RGB", (8, 8), (index, 0, 0)).save(path)
        rows.append({"frame_id": frame_id, "label_text": text, "path": str(path), "sha256": _sha256_file(path),
                     "view": "front"})
    value = {"schema_version": "website_assembly_coverage.v1", "label_readings": rows, "selected_frames": [],
             "body_bounds": dict(body) if body else None}
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value


def _research(tmp_path, website, invoker, *, coverage=None, context=OWNER, kind="revolute", root="spec"):
    return research.research_object_spec(target_id="dishwasher-1", category="dishwasher", articulation_kind=kind,
        coverage=coverage if coverage is not None else _coverage(tmp_path), task_context=context,
        output_root=tmp_path / root, invoker=invoker, transport=_transport(website["fetch"]), resolver=_public)


def test_only_figures_the_fetched_page_states_are_kept(tmp_path, website):
    invoker = _Invoker(_findings())
    record = _research(tmp_path, website, invoker)
    assert record["schema_version"] == "website_object_spec.v1" and record["status"] == "researched"
    assert record["digest"] == canonical_digest(record, digest_field="digest")
    specs = record["specs"]
    assert specs["overall_width"] == {"value": 0.598, "unit": "m", "source_urls": [SPEC], "match": "exact_model"}
    assert specs["overall_height"]["value"] == pytest.approx([33.875 * 0.0254, 0.8763])  # Sources disagree: range.
    assert specs["overall_height"]["source_urls"] == [RETAIL, SPEC]
    assert specs["overall_depth"]["value"] == pytest.approx(0.5715)
    assert {row["name"]: row["reason"] for row in record["unsourced_dropped"]} == {
        "net_weight": "quote_value_mismatch", "door_weight": "source_not_fetched",
        "upper_rack_max_load": "quote_not_in_page", "drawer_max_load": "quote_not_in_page"}
    assert record["silent_on"] == ["cutout_width", "cutout_height", "cutout_depth", "net_weight", "door_weight"]
    assert record["blockers"] == [] and record["dimension_check"]["blockers"] == []
    assert record["claim"] == research.CLAIM and record["physical_measurement_proven"] is False
    fetched = record["research"]["fetch_log"]
    assert [row["status"] for row in fetched] == ["ok", "ok"]
    assert fetched[1]["final_url"] == "https://www.retailer.example/p2" and fetched[0]["sha256"].startswith("sha256:")
    assert record["research"]["fetch_log_digest"].startswith("sha256:")
    assert record["research"]["receipt_binding_digest"].startswith("sha256:")
    assert record["research"]["agent_reported_silent_on"] == ["cutout_width"]
    # One bounded agent run: web search plus fetch_page, typed output, multi-turn.
    (spec, input_value), = invoker.calls
    assert spec.max_turns == research.MAX_TURNS and spec.output_type is research.ObjectSpecFindings
    assert [binding.tool_id for binding in spec.tool_bindings] == ["fetch_page"]
    assert type(spec.hosted_tools[0]).__name__ == "WebSearchTool"
    assert spec.max_hosted_tool_calls_per_turn == research.MAX_WEB_SEARCHES_PER_TURN
    assert sum(part["type"] == "input_image" for part in input_value[0]["content"]) == 1
    # Reserved against the scene cap before the call, settled to the charge after.
    (reservation,) = website["reserve"]
    assert reservation["resource_class"] == "openai_api_candidate" and reservation["provider"] == "openai"
    assert reservation["maximum_cost_usd"] == research.MAX_COST_USD
    assert [row["provider_charge_amount_usd"] for row in website["settle"]] == [0.21]


def test_owner_statement_outranks_label_text(tmp_path, website):
    invoker = _Invoker(_findings())
    record = _research(tmp_path, website, invoker, context=_context(item_make_model="Bosch SHPM88Z75N",
                                                                    item_weight="about 80 lb"))
    identity = record["identity"]
    assert identity["basis"] == "owner_stated" and identity["specificity"] == "brand_and_model"
    assert identity["owner_stated"] == {"item_make_model": "Bosch SHPM88Z75N", "item_weight": "about 80 lb"}
    assert identity["label_reads"] == [{"text": "800 Series", "frame_ids": ["decoded-000000001"]},
                                       {"text": "BOSCH", "frame_ids": ["decoded-000000001"]}]
    data = json.loads(invoker.calls[0][1][0]["content"][0]["text"].removeprefix("Product identity (data): "))
    assert data["owner_stated_make_model"] == "Bosch SHPM88Z75N" and data["identity_names_a_model"] is True
    assert "net_weight" not in record["specs"]  # An owner's weight is a statement, never a published figure.


def test_brand_read_from_a_label_cannot_claim_an_exact_model(tmp_path, website):
    record = _research(tmp_path, website, _Invoker(_findings()), context=_context(),
                       coverage=_coverage(tmp_path, labels=(("decoded-000000002", ["BOSCH"]),)))
    assert record["identity"]["basis"] == "label_read" and record["identity"]["specificity"] == "brand_only"
    assert record["product"]["match"] == "model_family"
    assert record["product"]["match_downgraded_from"] == "exact_model"
    assert {spec["match"] for spec in record["specs"].values()} == {"model_family"}
    exact = _research(tmp_path, website, _Invoker(_findings()), context=_context(), root="exact",
                      coverage=_coverage(tmp_path, labels=(("decoded-000000002", ["BOSCH", "SHPM88Z75N"]),)))
    assert exact["product"]["match"] == "exact_model" and "match_downgraded_from" not in exact["product"]


@pytest.mark.parametrize("env,context,labels,reason", [
    ({}, OWNER, (), "agent_disabled"),
    ({research.ENABLE_ENV: "1"}, OWNER, (), "live_agents_sdk_operators_not_allowed"),
    ({research.ENABLE_ENV: "1", "BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS": "1"}, _context(), (), "identity_unknown"),
])
def test_closed_gate_or_unknown_identity_is_not_run_and_spends_nothing(tmp_path, website, monkeypatch,
                                                                     env, context, labels, reason):
    for name in (research.ENABLE_ENV, "BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    invoker = _Invoker(_findings())
    record = _research(tmp_path, website, invoker, context=context, coverage=_coverage(tmp_path, labels=labels))
    assert record["status"] == "not_run" and record["research"] == {"status": "not_run", "reason": reason}
    assert record["specs"] == {} and record["blockers"] == []  # Not a blocker: estimates stay estimates.
    assert invoker.calls == website["reserve"] == website["fetch"] == []


def test_attach_keeps_matching_records_and_restart_buys_nothing(tmp_path, website):
    invoker = _Invoker(_findings())
    coverage = _coverage(tmp_path)
    cup = {"target_id": "cup-1", "authoring_coverage": coverage}
    uncovered = {"target_id": "oven-1", "articulation_kind": "revolute"}
    masks = {"targets": [{"target_id": "dishwasher-1", "articulation_kind": "revolute", "semantic_label": "dishwasher",
                          "authoring_coverage": coverage}, cup, uncovered], "digest": "sha256:" + "b" * 64}
    arguments = dict(removal_manifest={"entries": []}, task_context=OWNER, output_root=tmp_path / "spec",
                     invoker=invoker, transport=_transport(website["fetch"]), resolver=_public)
    value = research.attach_object_specs(task_masks=masks, **arguments)
    assert value["digest"] == canonical_digest(value, digest_field="digest") != masks["digest"]
    assert value["targets"][0]["object_spec"]["status"] == "researched"
    assert value["targets"][1:] == [cup, uncovered]
    assert research.attach_object_specs(task_masks=value, **arguments) == value
    # A lost record is rebuilt from the retained receipt, not bought again.
    assert research.attach_object_specs(task_masks=masks, **arguments) == value
    assert len(invoker.calls) == len(website["reserve"]) == len(website["settle"]) == 1
    assert len(website["fetch"]) == 3


def test_uncertain_receipt_is_held_for_reconciliation_and_never_rebought(tmp_path, website):
    invoker = _Invoker(_findings())
    _research(tmp_path, website, invoker)
    receipt = next((tmp_path / "spec" / "dishwasher-1").glob("research-*[0-9a-f].json"))
    receipt.write_text(json.dumps({"status": "submitting"}))
    record = _research(tmp_path, website, invoker)
    assert record["status"] == "held" and record["specs"] == {}
    assert record["blockers"] == ["website_object_spec_agent_requires_reconciliation"]
    assert len(invoker.calls) == len(website["reserve"]) == 1


def test_failed_agent_run_leaves_an_uncertain_receipt(tmp_path, website):
    class Failing(_Invoker):
        def invoke(self, spec, input_value):
            self.calls.append(spec)
            raise TimeoutError("wall clock")
    invoker = Failing(_findings())
    assert _research(tmp_path, website, invoker)["blockers"] == ["website_object_spec_agent_failed:TimeoutError"]
    assert _research(tmp_path, website, invoker)["blockers"] == ["website_object_spec_agent_requires_reconciliation"]
    assert len(invoker.calls) == len(website["reserve"]) == 1 and website["settle"] == []


def test_unknown_model_pricing_fails_closed_before_any_reservation(tmp_path, website, monkeypatch):
    monkeypatch.setattr(research, "MODEL", "gpt-unpriced")
    invoker = _Invoker(_findings())
    record = _research(tmp_path, website, invoker)
    assert record["status"] == "held" and record["blockers"] == ["website_object_spec_agent_pricing_unknown"]
    assert invoker.calls == website["reserve"] == []


def test_published_dimension_disagreeing_with_the_measured_body_blocks(tmp_path, website):
    short = {**BODY, "height_m": 0.4}
    record = _research(tmp_path, website, _Invoker(_findings()), coverage=_coverage(tmp_path, body=short))
    check = record["dimension_check"]
    assert check["comparisons"]["height"]["within_tolerance"] is False
    assert check["comparisons"]["width"]["within_tolerance"] is True
    assert record["blockers"] == ["website_object_spec_dimension_conflict"]
    family = _research(tmp_path, website, _Invoker(_findings(match="model_family")), root="family",
                       coverage=_coverage(tmp_path, body=short))
    assert family["dimension_check"]["comparisons"]["height"]["within_tolerance"] is False
    assert family["blockers"] == []  # A family figure that disagrees is reported, never a blocker.
    assert research.dimension_check({"specs": {"overall_height": {"value": [0.8, 0.9], "match": "exact_model"}}},
                                    {"height_m": 0.85})["comparisons"]["height"]["ratio"] == 1.0


def test_fetch_page_is_https_public_and_bounded(tmp_path, monkeypatch):
    calls = []
    pages = {"https://big.example/": {"status": 200, "headers": {"content-type": "text/html"}, "body": b"x" * 65},
             "https://down.example/": {"status": 302, "headers": {"location": "http://down.example/plain"}, "body": b""},
             "https://pdf.example/": {"status": 200, "headers": {"content-type": "application/pdf"}, "body": b"%PDF"},
             "https://ok.example/": {"status": 200, "headers": {"content-type": "text/plain"}, "body": b"Width 24 in"}}

    def transport(url, *, max_bytes, timeout):
        calls.append(url)
        return pages[url]

    def resolver(host, port, proto=0):
        return [(2, 1, 6, "", ("10.0.0.7" if host == "internal.example" else "93.184.216.34", port))]
    monkeypatch.setitem(sys.modules, "pypdf", None)  # No PDF extractor installed.
    fetcher = research.PageFetcher(pages_root=tmp_path, transport=transport, resolver=resolver, max_fetches=6,
                                   max_bytes=64)
    assert fetcher.fetch("http://ok.example/")["reason"] == "https_only"
    assert fetcher.fetch("https://internal.example/")["reason"] == "host_not_public"
    assert fetcher.fetch("https://big.example/")["reason"] == "page_too_large"
    assert fetcher.fetch("https://down.example/")["reason"] == "https_only"  # Redirect off HTTPS.
    assert fetcher.fetch("https://pdf.example/")["reason"] == "pdf_text_extraction_unavailable"
    assert fetcher.fetch("https://ok.example/")["excerpt"] == "Width 24 in"
    assert fetcher.fetch("https://ok.example/")["reason"] == "fetch_limit"
    assert calls == ["https://big.example/", "https://down.example/", "https://pdf.example/", "https://ok.example/"]
    assert [row["status"] for row in fetcher.log] == ["refused"] * 4 + ["unsupported", "ok", "refused"]
    assert all(row["fetched_at"] for row in fetcher.log)


def test_quoted_numbers_normalize_units_and_fractions():
    assert research.quoted_values("Width: 23½″ (59.8 cm)", default_unit="in") == pytest.approx(
        [23.5 * 0.0254, 0.598])
    assert research.quoted_values("Net weight 1,200 g", default_unit="kg") == pytest.approx([1.2])
    assert research.quoted_values("Load 10 kg / 22 lbs", default_unit="kg") == pytest.approx([10, 22 * 0.45359237])
    assert research.quoted_values("Height 34 inside", default_unit="in") == pytest.approx([34 * 0.0254])


def test_real_harness_reserves_and_reconciles_hosted_search_fees(tmp_path, monkeypatch):
    from agents.usage import Usage

    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS", "1")
    monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)
    seen = {}

    def run(agent, value, **kwargs):
        seen.update(agent=agent, kwargs=kwargs)
        return SimpleNamespace(final_output=research.ObjectSpecFindings.model_validate(_findings([])),
                               context_wrapper=SimpleNamespace(usage=Usage(requests=2, input_tokens=9000,
                                                                           output_tokens=800, total_tokens=9800)),
                               raw_responses=[], new_items=[SimpleNamespace(raw_item={"type": "web_search_call"})] * 3)
    invoker = research._default_invoker()
    invoker._run_agent = run
    reservations = []
    invoker.configure_reservation_audit(record_reservation=reservations.append, record_completion=lambda _: None,
                                        restored_reserved_cost_usd=0.0)
    fetcher = research.PageFetcher(pages_root=tmp_path, transport=_transport([]), resolver=_public)
    coverage = _coverage(tmp_path)
    frames = research._research_frames(coverage)
    identity = research.identify_object(task_context=OWNER, coverage=coverage, category="dishwasher")
    result = invoker.invoke(research._agent_spec("sha256:" + "d" * 64, fetcher),
                            research._agent_input(identity, "revolute", frames))
    (reservation,) = reservations
    assert reservation["hosted_tool_call_ceiling"] == research.MAX_TURNS * research.MAX_WEB_SEARCHES_PER_TURN
    assert reservation["projected_max_cost_usd"] <= research.MAX_COST_USD
    assert result.usage["hosted_tool_calls"] == 3
    token_cost = (9000 * 2.0 + 800 * 10.0) / 1_000_000
    assert result.cost_usd == pytest.approx(token_cost + 3 * research.WEB_SEARCH_USD_PER_CALL)
    assert seen["agent"].model_settings.extra_args["max_tool_calls"] == research.MAX_WEB_SEARCHES_PER_TURN
    assert [type(tool).__name__ for tool in seen["agent"].tools] == ["FunctionTool", "WebSearchTool"]
    assert seen["kwargs"]["max_turns"] == research.MAX_TURNS


def test_harness_refuses_hosted_tools_without_a_per_call_budget(tmp_path, monkeypatch):
    from dataclasses import replace

    from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import AgentsSDKInvocationBlocked
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS", "1")
    invoker = research._default_invoker()
    invoker._run_agent = lambda *args, **kwargs: pytest.fail("no provider call without a hosted-tool budget")
    fetcher = research.PageFetcher(pages_root=tmp_path, transport=_transport([]), resolver=_public)
    spec = replace(research._agent_spec("sha256:" + "d" * 64, fetcher), hosted_tool_call_usd=0.0)
    with pytest.raises(AgentsSDKInvocationBlocked, match="agents_sdk_hosted_tool_budget_missing"):
        invoker.invoke(spec, [{"role": "user", "content": [{"type": "input_text", "text": "x"}]}])
