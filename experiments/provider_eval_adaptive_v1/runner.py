"""Bounded adaptive research, with the diagnostic run's aggregate spend journal."""

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import ssl
import time
import urllib.request

from blueprint_pipeline.paid_resource_admission import PAID_LANE_ADMISSION_SCHEMA_VERSION, require_paid_resource_admission
from experiments.provider_eval_recovery.harness import Ledger, digest, exclusive, read_json, write_once
from experiments.provider_eval_recovery.live_http import (COUNT_ALLOWANCE, COUNT_ENDPOINT, ENDPOINTS,
    EXTRAS_PER_SEARCH_ATTEMPT, MAX_RESPONSE_BYTES, MODEL, NoRedirect, PROJECT, existing_key, credential_presence)
from experiments.provider_eval_recovery.live_runner import CATALOG, OWNER
from experiments.provider_eval_recovery.public_inputs import DECLARED_ORIGINAL_SHA256, load_public

from .protocol import (APPROVAL, CURRENT_DATE, LIMITS, MODES, PROTOCOL, RATES, ROOT, SEEDS,
    budget, code_hash, decision, evidence, model_envelopes, model_input, model_reserve, search_request)


class Blocked(RuntimeError):
    """Fixed sanitized transport/admission identifier."""


class CellStop(RuntimeError):
    """A bounded protocol abstention, not a factual grade."""


def admission(root, access, inputs):
    root = Path(root).resolve()
    public, _, provenance = load_public(inputs)
    if not provenance["byte_parity_with_original_public"]:
        raise Blocked("frozen_real_twenty_cases_required")
    if (access != read_json(root / "live_access.json") or access.get("status") != "existing_access_configured"
            or set(access.get("allowed_hosts", [])) != {"api.parallel.ai", "api.perplexity.ai", "api.openai.com"}
            or not (access.get("parallel_x_api_key_supported") is True
                    or (access.get("parallel_x_api_key_supported") == "unverified"
                        and access.get("parallel_header_pilot_probe_authorized") is True))
            or access.get("execution_owner_task_id") != OWNER
            or access.get("catalog_version") != CATALOG or access.get("openai_project") != PROJECT
            or access.get("verified_model") != MODEL or access.get("openai_model_http_status") != 200
            or access.get("journal_root") != str(root) or not all(credential_presence().values())):
        raise Blocked("existing_fresh_owner_bindings_required")
    old_scope = read_json(root / "live_scope.json")
    if (old_scope.get("budget_usd") != "10.00" or old_scope.get("model") != MODEL
            or old_scope.get("public_inputs_sha256") != DECLARED_ORIGINAL_SHA256
            or old_scope.get("journal_root") != str(root)):
        raise Blocked("original_aggregate_scope_required")
    plan = {"protocol": PROTOCOL, "approval": APPROVAL, "budget_usd": "10.00",
            "journal_root": str(root), "original_scope_sha256": digest(old_scope),
            "access_sha256": digest(access), "code_sha256": code_hash(), "model": MODEL,
            "project": PROJECT, "current_date": CURRENT_DATE, "public_inputs_sha256": DECLARED_ORIGINAL_SHA256,
            "public_semantic_sha256": digest(public), "seeds": list(SEEDS), "limits": {k: list(v) for k, v in LIMITS.items()},
            "max_searches_per_cell": 2, "transport_retries": 0, "page_retrieval": False,
            "incremental_ceiling_usd": str(budget()["incremental_usd"]),
            "review": "20 isolated four-output case reviews; no controller self-grading"}
    paths = root / "protocols" / PROTOCOL
    with exclusive(root):
        ledger = Ledger(root / "live_journal.jsonl", "10.00")
        guard(ledger, plan)
        write_once(paths / "scope.json", plan)
    return plan, public, paths


def guard(ledger, plan):
    ours = {event["attempt_id"] for event in ledger.events if event["kind"] == "reserved"
            and event.get("protocol") == PROTOCOL and event.get("plan_sha256") == digest(plan)}
    own_exposure = sum((ledger.reservations[key] for key in ours if ledger.states[key] != "not_accepted"), Decimal(0))
    if ledger.exposure - own_exposure + budget()["incremental_usd"] > Decimal("10"):
        raise Blocked("full_twenty_by_four_plus_independent_review_does_not_fit_no_reduced_matrix_selected")


class Transport:
    def __init__(self, root, access, public, plan, paths, *, opener=None):
        self.root, self.access, self.public, self.plan, self.paths = Path(root).resolve(), access, public, plan, paths
        if self.root != Path(plan["journal_root"]).resolve() or paths != self.root / "protocols" / PROTOCOL:
            raise Blocked("canonical_existing_aggregate_journal_required")
        self.opener = opener

    def retained(self, cell, step):
        ledger = Ledger(self.root / "live_journal.jsonl", "10.00")
        rows = [e for e in ledger.events if e["kind"] == "reserved" and e.get("protocol") == PROTOCOL
                and e.get("plan_sha256") == digest(self.plan) and e.get("cell") == cell and e.get("step") == step]
        if not rows:
            return None
        if len(rows) != 1 or ledger.states[rows[0]["attempt_id"]] != "completed":
            raise Blocked("uncertain_adaptive_attempt_no_retry")
        key = rows[0]["attempt_id"]
        retained = read_json(self.paths / "raw" / (key + ".json"))
        completed = next(e for e in ledger.events if e["kind"] == "completed" and e["attempt_id"] == key)
        if digest(retained) != completed["raw_sha256"] or digest(retained["request"]) != rows[0]["request_sha256"]:
            raise Blocked("retained_adaptive_evidence_integrity_failure")
        return retained

    def _send(self, provider, cell, step, envelope, reserve):
        # Sole private dispatch primitive: public methods construct every request.
        cells = {f"{i:02d}_{mode}" for i in range(1, 21) for mode in MODES}
        review_cells = {f"{i:02d}_review" for i in range(1, 21)}
        steps = {"search1", "search2", "assess_count", "assess", "synthesis_count", "synthesis", "review_count", "review"}
        reviewing = step in {"review_count", "review"}
        if (step not in steps or (cell not in review_cells if reviewing else cell not in cells)
                or (provider != "openai" and (step not in {"search1", "search2"} or not cell[3:].startswith(provider + "_")))
                or (provider == "openai" and step in {"search1", "search2"})):
            raise Blocked("frozen_adaptive_cell_step_provider_required")
        if envelope["method"] != "POST" or envelope["url"] not in {ENDPOINTS[provider], COUNT_ENDPOINT if provider == "openai" else ENDPOINTS[provider]}:
            raise Blocked("adaptive_endpoint_or_method_refused")
        binding = digest(self.plan)
        resource = "openai_api_candidate" if provider == "openai" else "evaluator_api"
        grant = require_paid_resource_admission(
            {"schema_version": PAID_LANE_ADMISSION_SCHEMA_VERSION, "status": "admitted", "resource_class": resource,
             "blockers": [], "allocation_binding_digest": binding},
            resource_class=resource, expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
        from blueprint_pipeline.paid_resource_admission import require_paid_resource_admission_grant
        require_paid_resource_admission_grant(grant, resource_class=resource,
                                             allocation_binding_digest=binding, require_allocation_binding=True)
        key = digest({"plan": binding, "cell": cell, "step": step, "request": envelope})
        with exclusive(self.root):
            if (read_json(self.paths / "scope.json") != self.plan or self.plan["code_sha256"] != code_hash()
                    or digest(read_json(self.root / "live_scope.json")) != self.plan["original_scope_sha256"]
                    or digest(read_json(self.root / "live_access.json")) != self.plan["access_sha256"]):
                raise Blocked("frozen_adaptive_scope_or_code_changed")
            ledger = Ledger(self.root / "live_journal.jsonl", "10.00")
            guard(ledger, self.plan)
            saved = self.retained(cell, step)
            if saved is not None:
                if saved["request"] != envelope:
                    raise Blocked("changed_request_no_adaptive_redispatch")
                return saved["raw"]
            secret = existing_key(provider)
            headers = {"Content-Type": "application/json"}
            headers["x-api-key" if provider == "parallel" else "Authorization"] = secret if provider == "parallel" else "Bearer " + secret
            if provider == "openai":
                headers["OpenAI-Project"] = PROJECT
            ledger.append("reserved", key, amount_usd=str(reserve), cell=cell, provider=provider,
                          role=PROTOCOL + ":" + step, step=step, protocol=PROTOCOL,
                          plan_sha256=binding, request_sha256=digest(envelope))
            try:
                started = time.monotonic()
                opener = self.opener
                if opener is None:
                    context = ssl.create_default_context(cafile=os.environ.get("CODEX_PROXY_CERT"))
                    opener = urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(context=context))
                req = urllib.request.Request(envelope["url"], data=json.dumps(envelope["body"]).encode(), headers=headers, method="POST")
                with opener.open(req, timeout=30) as response:
                    if response.status != 200:
                        raise Blocked("non_success")
                    blob = response.read(MAX_RESPONSE_BYTES + 1)
                if len(blob) > MAX_RESPONSE_BYTES:
                    raise Blocked("response_size")
                raw = json.loads(blob)
                if not isinstance(raw, dict) or (provider == "openai" and step.endswith("_count") and raw.get("object") != "response.input_tokens"):
                    raise Blocked("unexpected_response_shape")
                if provider == "openai" and not step.endswith("_count") and raw.get("model") != MODEL:
                    raise Blocked("returned_model_substitution")
                retained = {"request": envelope, "raw": raw, "latency_seconds": time.monotonic() - started,
                            "billing": "unreconciled; full reservation retained"}
                write_once(self.paths / "raw" / (key + ".json"), retained)
                ledger.append("completed", key, raw_sha256=digest(retained))
                return raw
            except Exception:
                ledger.append("uncertain", key, reason="adaptive_submission_or_response_uncertain")
                raise Blocked("uncertain_adaptive_submission_full_hold_no_retry") from None

    def search(self, index, mode, round_number, query):
        cell = f"{index:02d}_{mode}"
        case = self.public["cases"][index - 1]
        if round_number == 1:
            if query != SEEDS[index - 1]:
                raise Blocked("frozen_initial_keywords_required")
        elif round_number == 2:
            previous = self.retained(cell, "assess")
            if previous is None:
                raise Blocked("retained_controller_followup_required")
            check = decision(model_text(previous["raw"], "assess"), index, SEEDS[index - 1])
            if not check["needs_more"] or check["query"] != query:
                raise Blocked("followup_not_authorized_by_retained_coverage_check")
        else:
            raise Blocked("two_search_limit_no_transport_retries")
        envelope = search_request(index, mode, case, query)
        return self._send(mode.split("_")[0], cell, "search" + str(round_number), envelope,
                          RATES[mode] + EXTRAS_PER_SEARCH_ATTEMPT)

    def infer(self, cell, role):
        if role not in {"assess", "synthesis"}:
            raise Blocked("controller_role_or_oracle_access_refused")
        if cell not in {f"{i:02d}_{mode}" for i in range(1, 21) for mode in MODES}:
            raise Blocked("frozen_adaptive_cell_required")
        index, mode = int(cell[:2]), cell[3:]
        first = self.retained(cell, "search1")
        if first is None:
            raise Blocked("retained_initial_search_required")
        raws = [first["raw"]]
        if role == "synthesis":
            assessment = self.retained(cell, "assess")
            if assessment is None:
                raise Blocked("retained_coverage_check_required")
            check = decision(model_text(assessment["raw"], "assess"), index, SEEDS[index - 1])
            if check["needs_more"]:
                followup = self.retained(cell, "search2")
                if followup is None:
                    raise Blocked("retained_followup_required_before_synthesis")
                raws.append(followup["raw"])
        sources = evidence(mode, raws, 2000 if role == "assess" else 5500)
        items = model_input(role, self.public["cases"][index - 1], self.public["common_prompt"], sources,
                            query=SEEDS[index - 1] if role == "assess" else None)
        return self._infer_items(cell, role, items)

    def _infer_items(self, cell, role, items):
        count, response = model_envelopes(role, items)
        measured = self._send("openai", cell, role + "_count", count, COUNT_ALLOWANCE)
        tokens = measured.get("input_tokens")
        if type(tokens) is not int or not 0 < tokens <= LIMITS[role][0]:
            raise CellStop("input_token_budget_stop_no_recount")
        raw = self._send("openai", cell, role, response, model_reserve(role))
        usage = raw.get("usage", {})
        if usage.get("input_tokens") != tokens:
            raise Blocked("actual_input_usage_disagrees_with_count")
        return model_text(raw, role)


def model_text(raw, role):
    usage = raw.get("usage", {})
    if (raw.get("model") != MODEL or raw.get("service_tier", "default") != "default"
            or type(usage.get("input_tokens")) is not int or not 0 < usage["input_tokens"] <= LIMITS[role][0]
            or type(usage.get("output_tokens")) is not int or not 0 < usage["output_tokens"] <= LIMITS[role][1]):
        raise Blocked("actual_model_tier_or_usage_outside_adaptive_budget")
    text = "\n".join(part["text"] for item in raw.get("output", []) if item.get("type") == "message"
                     for part in item.get("content", []) if part.get("type") == "output_text")
    if raw.get("status") != "completed" or not text.strip():
        raise CellStop("output_token_budget_stop_no_retry")
    return text


def research(transport, index, mode):
    case, cell = transport.public["cases"][index - 1], f"{index:02d}_{mode}"
    check, sources, status, answer = None, [], "completed_ungraded", "Unknown: no supported answer within this protocol."
    try:
        raws = [transport.search(index, mode, 1, SEEDS[index - 1])]
        sources = evidence(mode, raws, 2000)
        check = decision(transport.infer(cell, "assess"), index, SEEDS[index - 1])
        if check["needs_more"]:
            raws.append(transport.search(index, mode, 2, check["query"]))
        sources = evidence(mode, raws, 5500)
        answer = transport.infer(cell, "synthesis")
    except CellStop as exc:
        status = "unknown_budget_stop:" + str(exc)
    except ValueError:
        status = "unknown_contract_or_evidence_stop_not_provider_quality"
    attempts = {}
    ledger = Ledger(transport.root / "live_journal.jsonl", "10.00")
    for event in ledger.events:
        if (event["kind"] == "reserved" and event.get("protocol") == PROTOCOL and event.get("plan_sha256") == digest(transport.plan)
                and event.get("cell") == cell and ledger.states[event["attempt_id"]] == "completed"):
            saved = transport.retained(cell, event["step"])
            attempts[event["attempt_id"]] = digest(saved)
    receipt = {"protocol": PROTOCOL, "case_id": case["id"], "cell": cell, "mode": mode,
               "status": status, "answer": answer, "sources": sources, "operational_coverage": check,
               "grade": "pending isolated reviewer; controller triage is not ground truth",
               "retained_attempts": attempts, "current_date": CURRENT_DATE}
    with exclusive(transport.root):
        write_once(transport.paths / "receipts" / (cell + ".json"), receipt)
    return receipt


def run(root, access, inputs, *, execute=False, owner_task_id=None, opener=None):
    plan, public, paths = admission(root, access, inputs)
    ledger = Ledger(Path(plan["journal_root"]) / "live_journal.jsonl", "10.00")
    if not execute:
        return {"status": "adaptive_preflight_no_network", "protocol": PROTOCOL,
                "existing_reserved_usd": str(ledger.exposure),
                "budget": {k: str(v) for k, v in budget().items()}, "matrix": "20cases x4modes",
                "diagnostic_answers_preserved": len(list((Path(root) / "live_receipts").glob("*.json")))}
    if owner_task_id != OWNER:
        raise Blocked("sole_fresh_execution_owner_required")
    transport = Transport(root, access, public, plan, paths, opener=opener)
    for index in range(1, 21):
        for mode in MODES:
            research(transport, index, mode)
    return {"status": "adaptive_outputs_complete_independent_review_pending", "protocol": PROTOCOL,
            "aggregate_reserved_usd": str(Ledger(transport.root / "live_journal.jsonl", "10.00").exposure)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="existing aggregate diagnostic journal root")
    parser.add_argument("--access-receipt", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=ROOT.parent / "provider_eval_recovery/real_public/inputs.parent-message.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execution-owner-task-id")
    args = parser.parse_args()
    try:
        result = run(args.output, read_json(args.access_receipt), args.input, execute=args.execute,
                     owner_task_id=args.execution_owner_task_id)
    except Blocked as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}))
        raise SystemExit(2) from None
    except (OSError, ValueError, KeyError, TypeError):
        print(json.dumps({"status": "blocked", "reason": "adaptive_metadata_or_retained_evidence_failed_closed"}))
        raise SystemExit(2) from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
