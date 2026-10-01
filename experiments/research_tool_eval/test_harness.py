"""Hermetic checks of dispatch, spend, evidence and provider contracts. No network."""
import asyncio
import json
import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from harness import (
    CONTEXT_CHARS,
    HARD_CASES,
    MAX_POLLS,
    RAW_MODES,
    RESPONSE_BYTES,
    ROOT,
    TASK_MODES,
    Adapter,
    ConfirmedRejected,
    GateError,
    Harness,
    Journal,
    MockController,
    MockTransport,
    UncertainWork,
    canonical,
    estimate,
    is_public_url,
    verify_caches,
    verify_freeze,
)


class Scripted(MockTransport):
    def __init__(self, actions):
        self.actions = list(actions)
        self.requests = []

    async def call(self, request, **kwargs):
        self.requests.append(request)
        if not self.actions:
            raise AssertionError("unexpected dispatch")
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        if action == "wait":
            await asyncio.sleep(10)
        return action


class Checks(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "journal.sqlite3"
        self.journal = Journal(self.path, cap_usd=10)
        self.case = json.loads((ROOT / "cases.json").read_text())[0]
        self.clock = time.time()

    def tearDown(self):
        self.journal.db.close()
        self.tmp.cleanup()

    def harness(self, transport=None, controller=None):
        return Harness(self.journal, transport or MockTransport(), controller or MockController(), clock=lambda: self.clock)

    async def test_complete_92_cells_no_network_or_paid_calls(self):
        h = self.harness()
        with patch.object(socket, "socket", side_effect=AssertionError("network prohibited")):
            cells = []
            for c in json.loads((ROOT / "cases.json").read_text()):
                for m in RAW_MODES:
                    cells.append(await h.run_case(c, m))
                if c["id"] in HARD_CASES:
                    for m in TASK_MODES:
                        cells.append(await h.run_case(c, m))
        self.assertEqual(92, len(cells))
        self.assertEqual(184, self.journal.totals()["logical_calls"])
        self.assertEqual(0, self.journal.totals()["actual_total_usd"])
        self.assertEqual(12, self.journal.totals()["polls"])
        self.assertTrue(all(x["grading"]["coverage"] is None for x in cells))

    async def test_timeout_unknown_not_resubmitted_after_restart(self):
        t = Scripted(["wait"])
        h = self.harness(t)
        req = Adapter("parallel-fast").request(self.case)
        req["timeout_s"] = .01
        with self.assertRaises(TimeoutError):
            await h.dispatch(req, .001)
        ident = h.call_id(req)
        self.assertEqual("uncertain", self.journal.row(ident)["state"])
        self.journal.db.close()
        self.journal = Journal(self.path, cap_usd=10)
        with self.assertRaises(UncertainWork):
            await self.harness(t).dispatch(req, .001)
        self.assertEqual(1, len(t.requests))
        self.assertIsNone(self.journal.totals()["actual_total_usd"])
        self.assertGreater(self.journal.totals()["reserved_upper_estimate_usd"], 0)

    async def test_crash_after_durable_intent_requires_reconciliation(self):
        h = self.harness()
        req = Adapter("perplexity-standard").request(self.case)
        ident = h.call_id(req)
        self.journal.reserve(ident, req, .005)
        self.journal.claim(ident)
        with self.assertRaises(UncertainWork):
            await h.dispatch(req, .005)

    async def test_concurrent_dispatch_claim_is_atomic(self):
        h = self.harness()
        req = Adapter("parallel-fast").request(self.case)
        ident = h.call_id(req)
        self.journal.reserve(ident, req, .001)
        second = Journal(self.path, cap_usd=10)
        self.journal.claim(ident)
        try:
            with self.assertRaises(UncertainWork):
                second.claim(ident)
        finally:
            second.db.close()

    async def test_confirmed_rejection_only_retry_is_bounded(self):
        t = Scripted([ConfirmedRejected(), ConfirmedRejected()])
        h = self.harness(t)
        req = Adapter("parallel-fast").request(self.case)
        for _ in range(2):
            with self.assertRaises(ConfirmedRejected):
                await h.dispatch(req, .001)
        with self.assertRaises(GateError):
            await h.dispatch(req, .001)
        self.assertEqual(2, len(t.requests))

    async def test_known_task_id_uses_get_on_resume_without_post(self):
        t = Scripted([{"run_id": "trun_123"}, {"run": {"status": "running"}},
                      {"run": {"status": "completed"}, "output": {"content": {}}, "usage": {"actual_usd": .025}}])
        h = self.harness(t)
        req = h.request_for(self.case, "parallel-core")
        await h.dispatch(req, .025)
        with self.assertRaises(UncertainWork):
            await h.run_case(self.case, "parallel-core")
        self.clock += 30
        cell = await h.run_case(self.case, "parallel-core")
        self.assertEqual(["POST", "GET", "GET"], [x["method"] for x in t.requests])
        self.assertEqual("delegated_research_system", cell["comparison_class"])

    async def test_task_poll_limit_survives_resume(self):
        t = Scripted([{"run_id": "trun_123"}] + [{"run": {"status": "running"}}] * MAX_POLLS)
        h = self.harness(t)
        req = h.request_for(self.case, "parallel-pro")
        await h.dispatch(req, .1)
        for _ in range(MAX_POLLS):
            with self.assertRaises(UncertainWork):
                await h.reconcile(req)
            self.clock += 30
        with self.assertRaises(GateError):
            await self.harness(t).reconcile(req)
        self.assertEqual(MAX_POLLS + 1, len(t.requests))

    async def test_failed_task_cannot_be_synthesized_or_refunded(self):
        t = Scripted([{"run_id": "trun_123"}, {"run": {"status": "failed"}}])
        h = self.harness(t)
        with self.assertRaises(UncertainWork):
            await h.run_case(self.case, "parallel-pro")
        self.assertEqual(.1, self.journal.totals()["reserved_upper_estimate_usd"])
        self.assertIsNone(self.journal.totals()["actual_total_usd"])

    async def test_cancellation_blocks_reserved_and_new_dispatches(self):
        h = self.harness()
        req = Adapter("parallel-fast").request(self.case)
        self.journal.reserve(h.call_id(req), req, .001)
        self.journal.cancel()
        with self.assertRaises(GateError):
            await h.dispatch(req, .001)
        with self.assertRaises(GateError):
            await h.run_case(self.case, "perplexity-fast")

    async def test_cancelled_run_can_get_already_accepted_task_result(self):
        t = Scripted([{"run_id": "trun_1"}, {"run": {"status": "completed"}, "output": {}, "usage": {"actual_usd": .025}}])
        h = self.harness(t)
        req = h.request_for(self.case, "parallel-core")
        await h.dispatch(req, .025)
        self.journal.cancel()
        result = await h.reconcile(req)
        self.assertEqual("completed", result["run"]["status"])
        self.assertEqual(["POST", "GET"], [x["method"] for x in t.requests])

    async def test_completed_replay_never_reanswers_or_retrieves(self):
        t = Scripted([{"results": [], "usage": {"actual_usd": 0}}])
        h = self.harness(t)
        first = await h.run_case(self.case, "parallel-fast")
        second = await h.run_case(self.case, "parallel-fast")
        self.assertEqual(first, second)
        self.assertEqual(1, len(t.requests))
        self.assertEqual(2, self.journal.totals()["dispatches"])

    async def test_provider_payloads_current_docs(self):
        p = Adapter("parallel-advanced").request(self.case)
        q = Adapter("perplexity-standard").request(self.case)
        self.assertEqual("/v1/search", p["path"])
        self.assertEqual(10, p["body"]["advanced_settings"]["max_results"])
        self.assertEqual("/search", q["path"])
        self.assertEqual("web", q["body"]["search_type"])
        self.assertEqual(p["body"]["search_queries"], q["body"]["query"])
        self.assertEqual(4000, q["body"]["max_tokens"])
        task = Adapter("parallel-core").request(self.case)
        self.assertEqual("/v1/tasks/runs", task["path"])
        self.assertEqual("json", task["body"]["task_spec"]["output_schema"]["type"])
        self.assertNotIn("idempotency_key", task["body"])

    async def test_blinded_controller_inputs_and_budgets_equal(self):
        class Capture(MockController):
            def __init__(self): self.inputs = []
            async def answer(self, request, **kw):
                self.inputs.append(request)
                return await super().answer(request, **kw)
        c = Capture()
        h = self.harness(controller=c)
        for m in RAW_MODES:
            await h.run_case(self.case, m)
        for request in c.inputs:
            self.assertNotIn("evaluation_cell", request)
            self.assertNotIn("mode", request["evidence"])
            self.assertNotIn("provider_usage", request["evidence"])
            self.assertEqual([], request["external_tools"])
            self.assertEqual(8000, request["max_input_tokens"])
            self.assertEqual(1600, request["max_output_tokens"])
            self.assertNotIn("expected_facts", canonical(request))

    async def test_retrieval_time_does_not_refresh_source_check_date(self):
        r = {"results": [{"url": "https://example.org/source", "snippet": "text", "date": "2022-01-01", "last_updated": "2023-01-01"}]}
        evidence = Adapter("perplexity-fast").normalize(r, retrieved_at="2026-09-30")
        self.assertIsNone(evidence["sources"][0]["source_checked_at"])
        self.assertEqual("2022-01-01", evidence["sources"][0]["published_at"])
        self.assertEqual("2026-09-30", evidence["sources"][0]["retrieved_at"])

    async def test_response_oversize_preserves_uncertain_paid_work(self):
        t = Scripted([{"results": [], "payload": "x" * (RESPONSE_BYTES + 1)}])
        h = self.harness(t)
        req = Adapter("parallel-fast").request(self.case)
        with self.assertRaises(ValueError):
            await h.dispatch(req, .001)
        self.assertEqual("uncertain", self.journal.row(h.call_id(req))["state"])
        with self.assertRaises(UncertainWork):
            await h.dispatch(req, .001)

    async def test_context_bounded_and_untrusted_urls_filtered(self):
        raw = {"results": [{"url": "https://example.org/" + str(i), "title": "x" * 10000, "snippet": "x" * 30000}
                           for i in range(10)] + [{"url": "https://app.notion.com/private", "snippet": "PRIVATE"}]}
        e = Adapter("perplexity-standard").normalize(raw, retrieved_at="2026-09-30")
        self.assertLessEqual(len(canonical(e["sources"])), CONTEXT_CHARS)
        self.assertNotIn("PRIVATE", canonical(e))
        deep = Adapter("parallel-core").normalize({"run": {"status": "completed"}, "output": {"content": "x" * 30000}}, retrieved_at="2026-09-30")
        self.assertEqual(CONTEXT_CHARS, len(deep["research_output_excerpt"]))
        self.assertTrue(deep["context_truncated"])

    async def test_cases_are_exact_and_private_extras_fail(self):
        for key, value in [("private_contact", "someone"), ("question", "contact CRM"), ("disclosure", "private")]:
            c = dict(self.case)
            c[key] = value
            with self.assertRaises(GateError):
                Adapter("parallel-fast").request(c)

    async def test_live_transport_or_controller_always_refused(self):
        t = MockTransport()
        t.offline = False
        with self.assertRaises(GateError):
            self.harness(t)
        c = MockController()
        c.offline = False
        with self.assertRaises(GateError):
            self.harness(controller=c)

    async def test_nonfinite_unknown_cost_and_budget_rejected(self):
        for cost in [float("nan"), float("inf"), -1, 0]:
            with self.assertRaises(GateError):
                self.journal.reserve(str(cost), {}, cost)
        with self.assertRaises(GateError):
            Journal(Path(self.tmp.name) / "bad.sqlite3", cap_usd=float("nan"))
        self.journal.reserve("one", {}, 1)
        with self.assertRaises(GateError):
            self.journal.update("one", actual=float("nan"))

    async def test_budget_calls_and_unexpected_actual_overrun_fail_closed(self):
        j = Journal(Path(self.tmp.name) / "small.sqlite3", cap_usd=.002, max_calls=2)
        try:
            j.reserve("a", {}, .001)
            j.reserve("b", {}, .001)
            with self.assertRaises(GateError):
                j.reserve("c", {}, .001)
            j.update("a", actual=.005)
            self.assertEqual(.006, j.totals()["reserved_upper_estimate_usd"])
            with self.assertRaises(GateError):
                j.claim("b")
            with self.assertRaises(GateError):
                j.reserve("d", {}, .001)
        finally:
            j.db.close()

    async def test_changed_budget_or_identity_refused_on_resume(self):
        self.harness()
        with self.assertRaises(GateError):
            Journal(self.path, cap_usd=100)
        c = MockController()
        c.snapshot = "changed-model"
        with self.assertRaises(GateError):
            self.harness(controller=c)

    async def test_provider_ids_reject_path_injection(self):
        for value in ["../secrets", "x?foo=1", "x/../y", ""]:
            with self.assertRaises(ValueError):
                Adapter("parallel-core").poll(value)

    async def test_frozen_case_and_source_integrity(self):
        self.assertEqual(64, len(verify_freeze()))
        cases = json.loads((ROOT / "cases.json").read_text())
        rubric = json.loads((ROOT / "rubric.json").read_text())
        sources = json.loads((ROOT / "source_evidence.json").read_text())
        self.assertEqual(20, len({x["id"] for x in cases}))
        self.assertEqual({x["id"] for x in cases}, {x["case_id"] for x in rubric})
        source_ids = {x["id"] for x in sources}
        for case in rubric:
            self.assertTrue(case["forbidden_inferences"])
            for fact in case["expected_facts"]:
                self.assertTrue(set(fact["source_ids"]) <= source_ids)
        for s in sources:
            self.assertEqual("retrieved", s["status"])
            self.assertTrue(s["anchor_verified"])
            self.assertLessEqual(s["quote_words"], 25)
        self.assertNotIn("docs.google.com", canonical(cases))
        self.assertNotIn("app.notion.com", canonical(cases))

    async def test_current_cost_estimates_do_not_promise_all_in_cap(self):
        self.assertEqual(2.8, estimate()["bounded_token_envelope_estimate_usd"])
        self.assertEqual(3.934, estimate(True)["bounded_token_envelope_estimate_usd"])
        self.assertFalse(estimate()["hard_all_in_cap"])
        self.assertFalse(estimate()["approved"])

    async def test_public_url_checks(self):
        for u in ["http://example.com", "https://localhost/a", "https://127.0.0.1/x", "https://[::1]/x",
                  "https://169.254.169.254/a", "https://docs.google.com/x", "https://app.notion.com/p/x", "https://user:pass@example.com/a"]:
            self.assertFalse(is_public_url(u), u)
        self.assertTrue(is_public_url("https://www.chefrobotics.ai/post/public"))

    async def test_cache_corruption_refused_and_missing_cache_explicit(self):
        root = Path(self.tmp.name) / "fixtures"
        root.mkdir()
        (root / "source_cache").mkdir()
        (root / "source_evidence.json").write_text('[{"id":"x","text_sha256":"bad"}]')
        (root / "provider_docs_manifest.json").write_text('[]')
        with patch("harness.ROOT", root):
            self.assertEqual(["x.txt"], verify_caches()["missing"])
            with self.assertRaises(GateError):
                verify_caches(require=True)
            (root / "source_cache" / "x.txt").write_text("tampered")
            with self.assertRaises(GateError):
                verify_caches()

    async def test_task_memory_scope_isolated_per_cell_and_run(self):
        h = self.harness()
        a = h.request_for(self.case, "parallel-core")["body"]["memory_scope_key"]
        b = h.request_for(self.case, "parallel-pro")["body"]["memory_scope_key"]
        self.assertNotEqual(a, b)
        self.assertEqual(a, self.harness().request_for(self.case, "parallel-core")["body"]["memory_scope_key"])
        other = Journal(Path(self.tmp.name) / "other.sqlite3", cap_usd=10)
        try:
            c = Harness(other, MockTransport(), MockController()).request_for(self.case, "parallel-core")["body"]["memory_scope_key"]
            self.assertNotEqual(a, c)
        finally:
            other.db.close()

    async def test_poll_cadence_and_deadline_are_durable(self):
        t = Scripted([{"run_id": "trun_1"}, {"run": {"status": "running"}}])
        h = self.harness(t)
        req = h.request_for(self.case, "parallel-core")
        await h.dispatch(req, .025)
        with self.assertRaises(UncertainWork):
            await h.reconcile(req)
        with self.assertRaises(UncertainWork):
            await self.harness(t).reconcile(req)
        self.assertEqual(2, len(t.requests))
        self.clock += 800
        with self.assertRaises(GateError):
            await h.reconcile(req)
        self.assertEqual(2, len(t.requests))


if __name__ == "__main__":
    unittest.main()
