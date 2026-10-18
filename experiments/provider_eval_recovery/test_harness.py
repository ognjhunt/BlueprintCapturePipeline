"""Hermetic contract checks. No API calls, credentials, billing, or deployment."""

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from .adapters import MODEL, MODES, Limits, live_send, normalize, public_case, request
from .harness import (ROOT, Ledger, cost_bounds, digest, load_frozen, query_replay, read_json,
                      run, sol_cost, write_once)
from .import_source import stage_zip
from .reviewer import grade, review_all


class OfflineContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.manifest, self.cases, self.fixtures = load_frozen(ROOT / "synthetic")
        self.oracle = read_json(ROOT / "synthetic/reviewer/oracle.json")
        self.case = self.cases[0]
        self.fixture = self.fixtures["S01_parallel_fast"]

    def replay_query(self, fixture, ledger=None, limits=Limits()):
        ledger = ledger or Ledger(self.output / "journal.jsonl", "25")
        result, calls = query_replay(ledger, self.output,
                                    request("parallel_fast", self.case, 0, limits),
                                    fixture, 0, limits, "cell", "parallel_fast")
        return result, calls, ledger

    def test_entire_twenty_case_matrix_is_offline_and_resumable(self):
        with patch.object(socket, "socket", side_effect=AssertionError("network forbidden")), \
                patch.object(socket, "create_connection", side_effect=AssertionError("network")):
            result = run(ROOT / "synthetic", self.output)
            reviews = review_all(ROOT / "synthetic", self.output)
            self.assertEqual(reviews["synthetic_claims_correct"], 160)
            self.assertEqual((result["completed_cells"], result["fixture_sends_this_invocation"]),
                             (80, 80))
            receipts = list((self.output / "receipts").glob("*.json"))
            self.assertEqual(len(receipts), 80)
            for path in receipts:
                receipt = read_json(path)
                self.assertEqual(receipt["cost"]["actual_external_usd"], "0.00")
                self.assertEqual(receipt["grade_status"], "awaiting_separate_reviewer_process")
                self.assertNotIn("grade", receipt)
                self.assertEqual(receipt["model"], MODEL)
            before = (self.output / "journal.jsonl").read_bytes()
            resumed = run(ROOT / "synthetic", self.output)
            self.assertEqual(resumed["fixture_sends_this_invocation"], 0)
            self.assertEqual(before, (self.output / "journal.jsonl").read_bytes())
            self.assertEqual(result["simulated_cost_exposure_usd"],
                             resumed["simulated_cost_exposure_usd"])

    def test_no_live_transport(self):
        with self.assertRaisesRegex(RuntimeError, "offline"):
            live_send()

    def test_public_boundary_rejects_oracle_and_crm(self):
        for key in ("oracle", "privateCRM", "expected_answer", "reviewer"):
            with self.assertRaisesRegex(ValueError, "schema"):
                public_case({**self.case, key: "PRIVATE_SENTINEL"})
        envelopes = [request(mode, self.case, 0) for mode in MODES]
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(envelopes))
        self.assertNotIn('"claims":', json.dumps(envelopes))

    def test_exact_current_modes_endpoints_and_identical_prompts(self):
        envelopes = [request(mode, self.case, 0) for mode in MODES]
        shared = envelopes[0]["body"]["objective"]
        for mode, envelope in zip(MODES, envelopes):
            if mode.startswith("parallel"):
                self.assertEqual(envelope["url"], "https://api.parallel.ai/v1/search")
                self.assertEqual(envelope["body"]["objective"], shared)
                self.assertEqual(envelope["body"]["search_queries"], [shared])
                self.assertEqual(envelope["body"]["client_model"], MODEL)
            else:
                self.assertEqual(envelope["url"], "https://api.perplexity.ai/search")
                self.assertEqual(envelope["body"]["query"], shared)
                self.assertEqual(envelope["body"]["search_type"],
                                 "fast" if mode.endswith("fast") else "web")
        self.assertEqual(len({e["timeout_seconds"] for e in envelopes}), 1)

    def test_uncertain_acceptance_holds_spend_and_never_resends(self):
        fixture = {"queries": [[{"outcome": "timeout", "latency_ms": 30000,
                                  "provider_id": None}, self.fixture["queries"][0][0]]]}
        raw, calls, ledger = self.replay_query(fixture)
        self.assertIsNone(raw)
        self.assertEqual(calls, 1)
        self.assertEqual(ledger.exposure, Decimal("0.001"))
        for _ in range(5):
            ledger = Ledger(self.output / "journal.jsonl", "25")
            raw, calls, ledger = self.replay_query(fixture, ledger)
            self.assertIsNone(raw)
            self.assertEqual(calls, 0)
        self.assertEqual(sum(ledger.reconciliations.values()), 2)
        self.assertEqual(len(ledger.reservations), 1)

    def test_crash_after_reservation_is_ambiguous_and_never_resends(self):
        ledger = Ledger(self.output / "journal.jsonl", "25")
        envelope = request("parallel_fast", self.case, 0)
        key = digest({"cell": "cell", "query": 0, "attempt": 1, "request": envelope})
        ledger.append("reserved", key, amount_usd="0.001", cell="cell")
        raw, calls, _ = self.replay_query(self.fixture, ledger)
        self.assertIsNone(raw)
        self.assertEqual(calls, 0)

    def test_retained_response_adoption_after_crash(self):
        raw, _, ledger = self.replay_query(self.fixture)
        # Simulate only the reserve durable; raw response installation succeeded
        # before the process could append the completed event.
        first_line = (self.output / "journal.jsonl").read_bytes().splitlines(keepends=True)[0]
        (self.output / "journal.jsonl").write_bytes(first_line)
        adopted, calls, ledger = self.replay_query(self.fixture)
        self.assertEqual(adopted, raw)
        self.assertEqual(calls, 0)
        self.assertEqual(ledger.exposure, Decimal("0.001"))
        self.assertEqual(ledger.events[-1]["synthetic_latency_ms"], raw["latency_ms"])

    def test_only_proven_nonacceptance_allows_one_bounded_retry(self):
        fixture = {"queries": [[{"outcome": "not_accepted", "latency_ms": 12},
                                 self.fixture["queries"][0][0]]]}
        raw, calls, ledger = self.replay_query(fixture)
        self.assertIsNotNone(raw)
        self.assertEqual(calls, 2)
        self.assertEqual(ledger.exposure, Decimal("0.001"))
        self.assertEqual([e["kind"] for e in ledger.events],
                         ["reserved", "not_accepted", "reserved", "completed"])
        self.assertEqual(sum(e.get("synthetic_latency_ms", 0) for e in ledger.events), 213)

    def test_rejected_attempts_exhaust_without_third_send(self):
        fixture = {"queries": [[{"outcome": "not_accepted", "latency_ms": 12}] * 2]}
        raw, calls, ledger = self.replay_query(fixture)
        self.assertIsNone(raw)
        self.assertEqual(calls, 2)
        self.assertEqual(ledger.exposure, 0)
        _, resumed_calls, _ = self.replay_query(fixture, ledger)
        self.assertEqual(resumed_calls, 0)

    def test_run_and_cell_cost_caps_before_dispatch(self):
        ledger = Ledger(self.output / "journal.jsonl", "0.0005")
        with self.assertRaisesRegex(ValueError, "run spend cap"):
            self.replay_query(self.fixture, ledger)
        self.assertEqual(len(ledger.events), 0)
        ledger = Ledger(self.output / "journal.jsonl", "25")
        with self.assertRaisesRegex(ValueError, "cell spend cap"):
            self.replay_query(self.fixture, ledger, replace(Limits(),
                              cell_cost_cap=Decimal("0.0005")))
        self.assertEqual(len(ledger.events), 0)

    def test_torn_and_tampered_ledger_fail_closed(self):
        self.replay_query(self.fixture)
        data = (self.output / "journal.jsonl").read_bytes()
        (self.output / "journal.jsonl").write_bytes(data[:-1])
        with self.assertRaisesRegex(ValueError, "torn"):
            Ledger(self.output / "journal.jsonl", "25")
        (self.output / "journal.jsonl").write_bytes(data.replace(b"0.001", b"0.009"))
        with self.assertRaisesRegex(ValueError, "integrity"):
            Ledger(self.output / "journal.jsonl", "25")

    def test_receipt_and_response_tampering_fail_closed(self):
        run(ROOT / "synthetic", self.output)
        receipt_path = next((self.output / "receipts").glob("*.json"))
        original = receipt_path.read_bytes()
        receipt = read_json(receipt_path)
        receipt["answer"]["claims"][0]["state"] = "unknown"
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "immutable"):
            run(ROOT / "synthetic", self.output)
        receipt_path.write_bytes(original)
        raw_path = next((self.output / "raw").glob("*.json"))
        raw = read_json(raw_path)
        raw["raw"]["results"][0]["title"] = "tampered"
        raw_path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "response integrity"):
            run(ROOT / "synthetic", self.output)

    def test_code_or_limits_change_cannot_resume_a_frozen_run(self):
        run(ROOT / "synthetic", self.output)
        with self.assertRaisesRegex(ValueError, "immutable"):
            run(ROOT / "synthetic", self.output, replace(Limits(), timeout_seconds=29))

    def test_interrupted_frozen_controller_reservation_can_resume_without_inference(self):
        from .harness import Ledger as RealLedger

        original = RealLedger.append

        def interrupt(ledger, kind, key, **fields):
            event = original(ledger, kind, key, **fields)
            if kind == "reserved" and fields.get("model") == MODEL and "query" not in fields:
                raise RuntimeError("controller reservation crash")
            return event

        with patch.object(RealLedger, "append", interrupt):
            with self.assertRaisesRegex(RuntimeError, "reservation crash"):
                run(ROOT / "synthetic", self.output)
        result = run(ROOT / "synthetic", self.output)
        self.assertEqual(result["completed_cells"], 80)
        self.assertEqual(result["fixture_sends_this_invocation"], 79)

    def test_common_evidence_cap_includes_titles_urls_and_all_rounds(self):
        limited = replace(Limits(), evidence_chars=120)
        source = self.fixture["queries"][0][0]["raw"]["results"][0]
        sources = normalize("parallel_fast", {"results": [source] * 20}, limited)
        self.assertLessEqual(len(sources), 10)
        self.assertLessEqual(sum(len(s["title"]) + len(s["url"]) + len(s["text"])
                                 for s in sources), 120)

    def test_unsupported_missing_citation_and_unknown_grading(self):
        sources = normalize("parallel_fast", self.fixture["queries"][0][0]["raw"])
        answer = json.loads(json.dumps(self.fixture["answer"]))
        answer["claims"][0]["citations"] = []
        result = grade(answer, sources, self.oracle["S01"])
        self.assertEqual(result["unsupported_assertions"], 1)
        self.assertTrue(result["rows"][1]["unknown_correct"])
        answer["claims"][1]["state"] = "supported"
        result = grade(answer, sources, self.oracle["S01"])
        self.assertEqual(result["unsupported_assertions"], 2)
        self.assertEqual(result["total"], 2)

    def test_url_match_without_entailment_is_insufficient(self):
        sources = normalize("parallel_fast", self.fixture["queries"][0][0]["raw"])
        sources[0]["text"] = "Unrelated marketing statement."
        result = grade(self.fixture["answer"], sources, self.oracle["S01"])
        self.assertTrue(result["rows"][0]["citation_resolves"])
        self.assertFalse(result["rows"][0]["state_correct"])

    def test_model_pin_token_cache_and_reasoning_cost_accounting(self):
        usage = self.fixture["controller_usage"]
        self.assertEqual(sol_cost(usage), Decimal("0.00642"))
        for model in ("gpt-6-sol", "gpt-6-luna", "gpt-6-astra"):
            with self.assertRaisesRegex(ValueError, "substitution"):
                sol_cost({**usage, "model": model})
        with self.assertRaisesRegex(ValueError, "token budget"):
            sol_cost({**usage, "output_tokens": 2049})
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            sol_cost({**usage, "reasoning_tokens": 401})
        with self.assertRaisesRegex(ValueError, "substitution"):
            sol_cost({**usage, "service_tier": "fast"})

    def test_all_in_cost_bounds_include_outer_sol_and_retries(self):
        bound = cost_bounds()
        self.assertEqual(Decimal(bound["raw_search_base_usd"]), Decimal("0.24"))
        self.assertEqual(Decimal(bound["raw_search_retry_ceiling_usd"]), Decimal("0.48"))
        self.assertEqual(Decimal(bound["subtotal_all_in_usd"]), Decimal("8.9952"))
        self.assertEqual(Decimal(bound["subtotal_with_extras_allowance_usd"]), Decimal("9.4952"))
        self.assertGreaterEqual(Decimal(bound["approved_total_cap_usd"]),
                                Decimal(bound["subtotal_all_in_usd"]))
        self.assertEqual(bound["search_request_attempt_cap"], 160)

    def test_reviewer_rejects_altered_or_duplicate_receipts_before_first_review(self):
        run(ROOT / "synthetic", self.output)
        path = self.output / "receipts/S01_parallel_fast.json"
        original = path.read_bytes()
        receipt = read_json(path)
        receipt["answer"]["claims"][0]["state"] = "unknown"
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "content failure"):
            review_all(ROOT / "synthetic", self.output)
        self.assertFalse((self.output / "reviews").exists())
        path.write_bytes(original)
        (self.output / "receipts/duplicate.json").write_bytes(original)
        with self.assertRaisesRegex(ValueError, "matrix identity"):
            review_all(ROOT / "synthetic", self.output)
        self.assertFalse((self.output / "reviews").exists())

    def test_pilot_and_remaining_adopt_once_and_share_cumulative_cap(self):
        pilot = run(ROOT / "synthetic", self.output, phase="pilot")
        self.assertEqual(pilot["completed_cells"], 8)
        repeated = run(ROOT / "synthetic", self.output, phase="pilot")
        self.assertEqual(repeated["fixture_sends_this_invocation"], 0)
        remaining = run(ROOT / "synthetic", self.output, phase="remaining")
        self.assertEqual(remaining["fixture_sends_this_invocation"], 72)
        self.assertEqual(remaining["cumulative_retained_cells"], 80)
        self.assertEqual(read_json(self.output / "plan.json")["run_cost_cap_usd"], "10")
        self.assertEqual(review_all(ROOT / "synthetic", self.output)["reviewed_cells"], 80)

    def test_exclusive_artifacts_never_overwrite(self):
        path = self.output / "receipt.json"
        write_once(path, {"a": 1})
        write_once(path, {"a": 1})
        with self.assertRaisesRegex(ValueError, "immutable"):
            write_once(path, {"a": 2})

    def test_source_import_verifies_bytes_and_preserves_partition(self):
        path = self.output / "source.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("public/inputs.json", '{"cases": []}')
            archive.writestr("reviewer/oracle.json", '{"PRIVATE_SENTINEL": true}')
            archive.writestr("reviewer/spec.md", "Private rubric.")
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        size = path.stat().st_size
        with self.assertRaisesRegex(ValueError, "SHA256"):
            stage_zip(path, self.output / "wrong")
        dest = self.output / "staged"
        result = stage_zip(path, dest, checksum, size)
        self.assertEqual(result["schema_integration"], "pending_spec_review")
        self.assertNotIn("PRIVATE_SENTINEL", (dest / "public/inputs.json").read_text())
        self.assertEqual((dest / "reviewer/spec.md").read_text(), "Private rubric.")

    def test_source_import_rejects_zip_path_traversal(self):
        path = self.output / "unsafe.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name, content in (("public/inputs.json", "{}"),
                                  ("reviewer/oracle.json", "{}"),
                                  ("reviewer/spec.md", "rubric"), ("../escape", "bad")):
                archive.writestr(name, content)
        with self.assertRaisesRegex(ValueError, "unsafe ZIP"):
            stage_zip(path, self.output / "staged", hashlib.sha256(path.read_bytes()).hexdigest(),
                      path.stat().st_size)

    def test_source_import_consumes_the_hashed_bytes_without_reopening(self):
        path = self.output / "source.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("public/inputs.json", '{"verified": true}')
            archive.writestr("reviewer/oracle.json", '{}')
            archive.writestr("reviewer/spec.md", "rubric")
        blob = path.read_bytes()
        real_zip = zipfile.ZipFile

        def mutate_original_on_archive_open(source, *args, **kwargs):
            path.write_bytes(b"replaced after verification")
            return real_zip(source, *args, **kwargs)

        with patch("experiments.provider_eval_recovery.import_source.zipfile.ZipFile",
                   mutate_original_on_archive_open):
            stage_zip(path, self.output / "staged", hashlib.sha256(blob).hexdigest(), len(blob))
        self.assertEqual(read_json(self.output / "staged/public/inputs.json"), {"verified": True})

    def test_provider_process_never_parses_reviewer_oracle(self):
        original = json.loads

        def no_oracle_load(content, *args, **kwargs):
            parsed = original(content, *args, **kwargs)
            if isinstance(parsed, dict) and "S01" in parsed:
                self.fail("provider/controller process parsed reviewer oracle")
            return parsed

        with patch("experiments.provider_eval_recovery.harness.json.loads", no_oracle_load):
            result = run(ROOT / "synthetic", self.output)
        self.assertEqual(result["completed_cells"], 80)


if __name__ == "__main__":
    unittest.main()
