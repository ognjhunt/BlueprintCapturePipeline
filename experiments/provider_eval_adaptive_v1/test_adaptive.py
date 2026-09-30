"""Hermetic adaptive protocol tests; no real keys, HTTP, invoices or oracle."""

from decimal import Decimal
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.provider_eval_recovery.harness import Ledger, write_once
from experiments.provider_eval_recovery.live_http import COUNT_ENDPOINT, MODEL, PROJECT
from experiments.provider_eval_recovery.live_runner import CATALOG, OWNER
from experiments.provider_eval_recovery.public_inputs import DECLARED_ORIGINAL_SHA256

from .protocol import ENTITIES, MODES, ROOT, SEEDS, budget, decision, evidence, model_input, search_request
from .reviewer import ReviewerTransport, bind_reviewer, review_case
from .runner import Blocked, Transport, admission, research, run


class Response(io.BytesIO):
    status = 200


class MockHTTP:
    def __init__(self):
        self.requests, self.warn, self.fail, self.large_count = [], False, False, False
        self.needs_more = True

    def open(self, req, timeout):
        self.requests.append(req)
        if self.fail:
            raise TimeoutError("PRIVATE_PROXY_ERROR")
        body = json.loads(req.data)
        if req.full_url == COUNT_ENDPOINT:
            raw = {"object": "response.input_tokens", "input_tokens": 1501 if self.large_count else 500}
        elif req.full_url.endswith("/responses"):
            data = json.loads(body["input"][1]["content"])
            if body["max_output_tokens"] == 512:
                index = int(data["public_case"]["id"][-2:])
                text = json.dumps({"needs_more": self.needs_more, "missing": ["limits"],
                                   "query": ENTITIES[index - 1] + " deployment evidence 2026" if self.needs_more else None})
            elif body["max_output_tokens"] == 1216:
                text = "Current site fit remains unknown. Public source: https://example.com/vendor"
            else:
                text = json.dumps({"arms": ["unknown"] * 4, "limitations": "SYNTHETIC_REVIEW_ONLY"})
            raw = {"model": MODEL, "status": "completed", "service_tier": "default",
                   "usage": {"input_tokens": 500, "output_tokens": 50},
                   "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}
        else:
            result = {"url": "https://example.com/vendor", "title": "Public vendor evidence"}
            result.update({"excerpts": ["Task and version documented; current site suitability unknown."]}
                          if "mode" in body else {"snippet": "Task and version documented; current site suitability unknown."})
            raw = {"results": [result], "warnings": [{"type": "input_validation", "message": "truncated"}] if self.warn else []}
        return Response(json.dumps(raw).encode())


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.inputs = ROOT.parent / "provider_eval_recovery/real_public/inputs.parent-message.json"
        self.access = {"status": "existing_access_configured", "allowed_hosts": ["api.parallel.ai", "api.perplexity.ai", "api.openai.com"],
                       "parallel_x_api_key_supported": "unverified", "parallel_header_pilot_probe_authorized": True,
                       "execution_owner_task_id": OWNER, "catalog_version": CATALOG, "openai_project": PROJECT,
                       "verified_model": MODEL, "openai_model_http_status": 200, "journal_root": str(self.root)}
        write_once(self.root / "live_access.json", self.access)
        write_once(self.root / "live_scope.json", {"budget_usd": "10.00", "model": MODEL,
                   "public_inputs_sha256": DECLARED_ORIGINAL_SHA256, "journal_root": str(self.root)})
        Ledger(self.root / "live_journal.jsonl", "10.00").append("reserved", "prior-diagnostic",
               amount_usd="2.285510", cell="10_parallel_advanced", provider="parallel", role="search")
        env = patch.dict(os.environ, {"PARALLEL_API_KEY": "mock-only", "PERPLEXITY_API_KEY": "mock-only", "OPENAI_API_KEY": "mock-only"})
        env.start()
        self.addCleanup(env.stop)
        self.opener = MockHTTP()

    def transport(self, reviewer=False):
        plan, public, paths = admission(self.root, self.access, self.inputs)
        cls = ReviewerTransport if reviewer else Transport
        return cls(self.root, self.access, public, plan, paths, opener=self.opener)

    def invoke(self, execute=False):
        return run(self.root, self.access, self.inputs, execute=execute, owner_task_id=OWNER, opener=self.opener)

    def test_exact_budget_and_all_twenty_entity_first_native_queries(self):
        self.assertEqual(budget()["incremental_usd"], Decimal("7.572000"))
        self.assertEqual(budget()["prior_max_usd"], Decimal("2.428000"))
        transport = self.transport()
        for index, case in enumerate(transport.public["cases"], 1):
            for mode in MODES:
                request = search_request(index, mode, case, SEEDS[index - 1])
                body = request["body"]
                if mode.startswith("parallel"):
                    self.assertLessEqual(len(body["search_queries"][0]), 200)
                    self.assertEqual(body["search_queries"], [SEEDS[index - 1]])
                    self.assertIn(case["question"], body["objective"])
                    self.assertNotIn(transport.public["common_prompt"], body["objective"])
                else:
                    self.assertTrue(body["query"].startswith(SEEDS[index - 1]))
                    self.assertIn(case["question"], body["query"])
                    self.assertNotIn("objective", body)

    def test_existing_secure_access_boundary_is_not_weakened(self):
        for changes in ({"status": "pending"}, {"allowed_hosts": ["api.openai.com"]},
                        {"parallel_x_api_key_supported": False}, {"parallel_header_pilot_probe_authorized": False}):
            invalid = {**self.access, **changes}
            (self.root / "live_access.json").write_text(json.dumps(invalid))
            with self.assertRaises(Blocked):
                admission(self.root, invalid, self.inputs)
        self.assertEqual(self.opener.requests, [])

    def test_full_worst_case_with_independent_reviews_shared_cap_and_no_replay(self):
        old_scope = (self.root / "live_scope.json").read_bytes()
        self.assertEqual(self.invoke()["status"], "adaptive_preflight_no_network")
        self.assertEqual(self.opener.requests, [])
        output = self.invoke(True)
        self.assertEqual(Decimal(output["aggregate_reserved_usd"]), Decimal("8.747910"))
        self.assertEqual(len(self.opener.requests), 480)
        self.invoke(True)
        self.assertEqual(len(self.opener.requests), 480)
        transport = self.transport(True)
        bundle = {"spec": "SYNTHETIC_REVIEW_ONLY", "cases": {c["id"]: {"truth": "SYNTHETIC_REVIEW_ONLY"} for c in transport.public["cases"]}}
        bind_reviewer(transport, bundle)
        for index, case in enumerate(transport.public["cases"], 1):
            review_case(transport, index, bundle["cases"][case["id"]], bundle["spec"])
        self.assertEqual(len(self.opener.requests), 520)
        self.assertEqual(Ledger(self.root / "live_journal.jsonl", "10.00").exposure, Decimal("9.857510"))
        self.assertEqual((self.root / "live_scope.json").read_bytes(), old_scope)
        for req in self.opener.requests[:480]:
            self.assertNotIn("SYNTHETIC_REVIEW_ONLY", req.data.decode())
        for req in self.opener.requests:
            if req.full_url.endswith("/responses"):
                self.assertIn("Trusted current date: 2026-09-30", json.loads(req.data)["input"][0]["content"])
        self.assertNotIn("mock-only", (self.root / "live_journal.jsonl").read_text())

    def test_insufficient_budget_never_selects_reduced_matrix_or_dispatches(self):
        ledger = Ledger(self.root / "live_journal.jsonl", "10.00")
        ledger.append("reserved", "additional-old-spend", amount_usd="0.15", cell="11_parallel_fast", provider="parallel", role="search")
        with self.assertRaisesRegex(Blocked, "no_reduced_matrix"):
            self.invoke(True)
        self.assertEqual(self.opener.requests, [])

    def test_full_first_round_does_not_crowd_out_distinct_followup_evidence(self):
        for mode in MODES:
            first = {"results": [{"url": f"https://example.com/initial-{i}", "title": "Initial",
                        "excerpts": ["Incomplete initial evidence"], "snippet": "Incomplete initial evidence"}
                                  for i in range(10)]}
            second = {"results": [{"url": "https://example.com/needed-primary", "title": "Needed primary",
                                  "excerpts": ["Followup supplies missing version"],
                                  "snippet": "Followup supplies missing version"}]}
            for chars in (1000, 5500):
                selected = evidence(mode, [first, second], chars)
                self.assertEqual(selected[0]["url"], "https://example.com/needed-primary")
                self.assertIn("Followup supplies missing version", selected[0]["text"])
                self.assertLessEqual(len(selected), 10)
                self.assertLessEqual(sum(len(s["url"]) + len(s["title"]) + len(s["text"]) for s in selected), chars)
                self.assertTrue(any("initial-" in s["url"] for s in selected))
                items = model_input("synthesis", {"id": "BP-EVAL-01"}, "Public grading instructions", selected)
                self.assertIn("Followup supplies missing version", json.dumps(items))

    def test_warning_stops_before_model_and_is_not_quality_grade(self):
        self.opener.warn = True
        receipt = research(self.transport(), 1, "parallel_fast")
        self.assertIn("not_provider_quality", receipt["status"])
        self.assertEqual(len(self.opener.requests), 1)
        self.assertEqual(receipt["operational_coverage"], None)

    def test_count_stop_no_recount_and_no_inference(self):
        self.opener.large_count = True
        receipt = research(self.transport(), 1, "perplexity_fast")
        self.assertIn("budget_stop", receipt["status"])
        self.assertEqual(len(self.opener.requests), 2)
        self.assertFalse(any(r.full_url.endswith("/responses") for r in self.opener.requests))

    def test_malformed_coverage_shape_is_value_error_not_matrix_halt(self):
        for raw in ("null", "[]", '{"needs_more":true,"query":"Chef Robotics deployment","missing":[{}]}'):
            with self.assertRaises(ValueError):
                decision(raw, 1, SEEDS[0])

    def test_uncertain_submission_preserves_hold_and_never_retries(self):
        transport = self.transport()
        self.opener.fail = True
        with self.assertRaisesRegex(Blocked, "uncertain"):
            research(transport, 1, "parallel_fast")
        with self.assertRaisesRegex(Blocked, "uncertain"):
            research(transport, 1, "parallel_fast")
        self.assertEqual(len(self.opener.requests), 1)
        self.assertEqual(Ledger(self.root / "live_journal.jsonl", "10.00").exposure, Decimal("2.289635"))

    def test_second_search_needs_retained_controller_decision_and_third_refused(self):
        transport = self.transport()
        with self.assertRaises(Blocked):
            transport.search(1, "parallel_fast", 2, "Chef Robotics deployment evidence 2026")
        with self.assertRaises(Blocked):
            transport.search(1, "parallel_fast", 3, SEEDS[0])
        self.assertEqual(self.opener.requests, [])
        self.opener.needs_more = False
        research(transport, 1, "parallel_fast")
        with self.assertRaises(Blocked):
            transport.search(1, "parallel_fast", 2, "Chef Robotics deployment evidence 2026")
        self.assertEqual(len(self.opener.requests), 5)

    def test_readonly_reviewer_cannot_start_missing_research_or_change_oracle(self):
        transport = self.transport(True)
        bundle = {"spec": "synthetic", "cases": {c["id"]: "synthetic" for c in transport.public["cases"]}}
        bind_reviewer(transport, bundle)
        with self.assertRaises((Blocked, FileNotFoundError)):
            review_case(transport, 1, "synthetic", "synthetic")
        with self.assertRaises(Blocked):
            review_case(transport, 1, "changed oracle", "synthetic")
        self.assertEqual(self.opener.requests, [])

    def test_scope_budget_guard_rechecks_other_protocol_spend_before_dispatch(self):
        transport = self.transport()
        Ledger(self.root / "live_journal.jsonl", "10.00").append("reserved", "new-other-spend", amount_usd="0.15",
               cell="12_parallel_advanced", provider="parallel", role="search")
        with self.assertRaisesRegex(Blocked, "no_reduced_matrix"):
            research(transport, 1, "parallel_fast")
        self.assertEqual(self.opener.requests, [])


if __name__ == "__main__":
    unittest.main()
