"""Hermetic end-to-end live pilot tests. All HTTP and keys are mocked."""

from decimal import Decimal
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .adapters import MODEL, normalize
from .harness import ROOT, Ledger, digest, read_json
from .live_http import COUNT_ENDPOINT, COUNT_METHOD, LiveBlocked, PROJECT
from .live_runner import (APPROVAL, CATALOG, OWNER, QUERY_PATCH_BASE_CODE, QUERY_RECOVERY_CELL,
                          reconcile_query, run)

BMW_ARTICLE = ("https://www.press.bmwgroup.com/global/article/detail/T0458778EN/"
               "bmw-group-advances-the-use-of-physical-ai-in-production-with-figure-03-project-in-spartanburg")
BMW_LANGUAGE_CITATION = BMW_ARTICLE + "?language=en"


class Response(io.BytesIO):
    status = 200


class MockProviders:
    def __init__(self):
        self.requests = []
        self.count = 500
        self.fail = False
        self.bmw_query_case10 = False

    def open(self, req, timeout):
        self.requests.append(req)
        if self.fail:
            raise TimeoutError("private proxy error")
        body = json.loads(req.data)
        if req.full_url == COUNT_ENDPOINT:
            raw = {"object": "response.input_tokens", "input_tokens": self.count}
        elif req.full_url.endswith("/responses"):
            raw = {"model": MODEL, "status": "completed", "service_tier": "default",
                   "usage": {"input_tokens": self.count, "output_tokens": 40,
                             "input_tokens_details": {"cached_tokens": 0},
                             "output_tokens_details": {"reasoning_tokens": 0}},
                   "output": [{"type": "message", "content": [{"type": "output_text",
                               "text": "Unknown site fit. Source: https://example.com/vendor"}]}]}
        else:
            text = "Primary vendor claim; no site qualification established."
            result = {"url": "https://example.com/vendor", "title": "Vendor source"}
            result.update({"excerpts": [text]} if "mode" in body else {"snippet": text})
            results = [result]
            if self.bmw_query_case10 and body.get("mode") == "advanced" and "Entity: Figure AI" in body.get("objective", ""):
                results = [dict(result) for _ in range(10)]
                results[-1]["url"] = BMW_LANGUAGE_CITATION
            raw = {"results": results}
        return Response(json.dumps(raw).encode())


class LivePilotCommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name).resolve()
        self.input = ROOT / "real_public/inputs.parent-message.json"
        self.access = {"status": "existing_access_configured", "openai_project": PROJECT,
                       "allowed_hosts": ["api.parallel.ai", "api.perplexity.ai", "api.openai.com"],
                       "parallel_x_api_key_supported": True, "tokenizer_id": COUNT_METHOD,
                       "execution_owner_task_id": OWNER, "catalog_version": CATALOG,
                       "budget_approval": APPROVAL,
                       "key_presence": {"parallel": True, "perplexity": True, "openai": True},
                       "openai_model_http_status": 200, "verified_model": MODEL,
                       "prior_spend_usd": "0.00", "prior_paid_calls": 0,
                       "journal_root": str(self.output)}
        env = patch.dict(os.environ, {"PARALLEL_API_KEY": "mock-only",
                                     "PERPLEXITY_API_KEY": "mock-only", "OPENAI_API_KEY": "mock-only"})
        env.start()
        self.addCleanup(env.stop)
        self.opener = MockProviders()

    def invoke(self, phase="pilot", **kwargs):
        return run(self.input, self.output, self.access, phase, opener=self.opener,
                   owner_task_id=OWNER, **kwargs)

    def test_preflight_has_no_dispatch_and_requires_exact_owner_configuration(self):
        self.assertEqual(self.invoke()["status"], "preflight_passed_no_network")
        self.assertEqual(self.opener.requests, [])
        self.access["catalog_version"] = "old-session"
        with self.assertRaises(LiveBlocked):
            self.invoke(execute=True)
        self.assertEqual(self.opener.requests, [])

    def test_pilot_then_remaining_resume_exactly_once_under_total_cap(self):
        pilot = self.invoke(execute=True)
        self.assertEqual(pilot["cells"], 8)
        self.assertEqual(len(self.opener.requests), 24)
        self.assertEqual(Decimal(pilot["cumulative_reserved_usd"]), Decimal("0.49284"))
        self.invoke(execute=True)
        self.assertEqual(len(self.opener.requests), 24)
        full = self.invoke("remaining", execute=True)
        self.assertEqual(full["cells"], 72)
        self.assertEqual(len(self.opener.requests), 240)
        self.assertEqual(Decimal(full["cumulative_reserved_usd"]), Decimal("4.92840"))
        self.invoke("remaining", execute=True)
        self.assertEqual(len(self.opener.requests), 240)
        receipts = list((self.output / "live_receipts").glob("*.json"))
        self.assertEqual(len(receipts), 80)
        wire = "\n".join(req.data.decode() for req in self.opener.requests)
        self.assertNotIn("oracle", wire)
        self.assertNotIn("privateCRM", wire)
        self.assertNotIn("mock-only", (self.output / "live_journal.jsonl").read_text())

    def test_failed_search_holds_spend_and_never_resubmits(self):
        self.opener.fail = True
        with self.assertRaisesRegex(LiveBlocked, "uncertain_submission"):
            self.invoke(execute=True)
        with self.assertRaisesRegex(LiveBlocked, "uncertain_existing_attempt"):
            self.invoke(execute=True)
        self.assertEqual(len(self.opener.requests), 1)
        self.assertGreater(Ledger(self.output / "live_journal.jsonl", "10.00").exposure, 0)

    def test_count_above_cap_or_unavailable_never_generates(self):
        self.opener.count = 6001
        with self.assertRaises(LiveBlocked):
            self.invoke(execute=True)
        self.assertEqual(len(self.opener.requests), 2)
        self.assertFalse(any(req.full_url.endswith("/responses") for req in self.opener.requests))

    def test_remaining_requires_untampered_complete_pilot(self):
        with self.assertRaisesRegex(LiveBlocked, "complete_pilot"):
            self.invoke("remaining", execute=True)
        self.assertEqual(self.opener.requests, [])

    def test_retained_tamper_blocks_remaining_before_dispatch(self):
        self.invoke(execute=True)
        path = next((self.output / "live_raw").glob("*.json"))
        blob = read_json(path)
        blob["latency_seconds"] = -1
        path.write_text(json.dumps(blob))
        with self.assertRaisesRegex(LiveBlocked, "integrity"):
            self.invoke("remaining", execute=True)
        self.assertEqual(len(self.opener.requests), 24)

    def test_missing_count_metadata_blocks_before_any_search(self):
        self.access.pop("tokenizer_id")
        with self.assertRaises(LiveBlocked):
            self.invoke()
        with self.assertRaises(LiveBlocked):
            self.invoke(execute=True)
        self.assertEqual(self.opener.requests, [])

    def test_forged_empty_pilot_receipts_cannot_admit_remaining(self):
        directory = self.output / "live_receipts"
        directory.mkdir()
        for index in range(1, 3):
            for mode in ("parallel_fast", "parallel_advanced", "perplexity_fast", "perplexity_standard"):
                (directory / f"{index:02d}_{mode}.json").write_text('{"retained_attempts":{}}')
        with self.assertRaisesRegex(LiveBlocked, "exact_completed"):
            self.invoke("remaining", execute=True)
        self.assertEqual(self.opener.requests, [])

    def test_changed_answer_in_pilot_receipt_blocks_remaining(self):
        self.invoke(execute=True)
        path = next((self.output / "live_receipts").glob("*.json"))
        receipt = read_json(path)
        receipt["answer"] = "Fabricated winner"
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(LiveBlocked, "content_integrity"):
            self.invoke("remaining", execute=True)
        self.assertEqual(len(self.opener.requests), 24)

    def test_unverified_proxy_header_can_only_be_tested_in_admitted_pilot(self):
        self.access["parallel_x_api_key_supported"] = "unverified"
        with self.assertRaises(LiveBlocked):
            self.invoke(execute=True)
        self.assertEqual(self.opener.requests, [])
        self.access["parallel_header_pilot_probe_authorized"] = True
        self.assertEqual(self.invoke(execute=True)["cells"], 8)
        self.assertEqual(len(self.opener.requests), 24)
        self.assertEqual(dict(self.opener.requests[0].header_items())["X-api-key"], "mock-only")

    def paused_original_run(self):
        self.opener.bmw_query_case10 = True

        def old_normalize(mode, raw):
            if any("?" in result["url"] for result in raw["results"]):
                raise ValueError("original normalizer rejected query")
            return normalize(mode, raw)

        with patch("experiments.provider_eval_recovery.live_runner.code_hash", return_value=QUERY_PATCH_BASE_CODE), \
                patch("experiments.provider_eval_recovery.live_http.normalize", side_effect=old_normalize):
            self.invoke(execute=True)
            with self.assertRaisesRegex(LiveBlocked, "uncertain_submission"):
                self.invoke("remaining", execute=True)
        ledger = Ledger(self.output / "live_journal.jsonl", "10.00")
        self.assertEqual(len(list((self.output / "live_receipts").glob("*.json"))), 37)
        self.assertEqual(ledger.exposure, Decimal("2.285510"))
        uncertain = [key for key, state in ledger.states.items() if state == "uncertain"]
        self.assertEqual(len(uncertain), 1)
        key = uncertain[0]
        return key, digest(read_json(self.output / "live_raw" / (key + ".json")))

    def recover(self, retained_sha256):
        return reconcile_query(self.input, self.output, self.access, "remaining",
                               cell=QUERY_RECOVERY_CELL, retained_sha256=retained_sha256,
                               owner_task_id=OWNER)

    def test_original_37_cell_run_adopts_bmw_query_without_search_or_extra_cost(self):
        key, sha = self.paused_original_run()
        scope_before = (self.output / "live_scope.json").read_bytes()
        reservations_before = Ledger(self.output / "live_journal.jsonl", "10.00").reservations
        self.assertEqual(len(self.opener.requests), 112)
        with patch("experiments.provider_eval_recovery.live_http.existing_key", side_effect=AssertionError("no secret reads")):
            recovered = self.recover(sha)
            self.assertEqual(self.recover(sha), recovered)
        self.assertEqual(recovered["provider_calls"], 0)
        self.assertEqual(recovered["attempt_id"], key)
        self.assertEqual(len(self.opener.requests), 112)
        adopted = Ledger(self.output / "live_journal.jsonl", "10.00")
        self.assertEqual(adopted.reservations, reservations_before)
        self.assertEqual(adopted.exposure, Decimal("2.285510"))
        self.assertEqual(adopted.states[key], "completed")
        self.assertEqual((self.output / "live_scope.json").read_bytes(), scope_before)
        final = self.invoke("remaining", execute=True)
        self.assertEqual(len(self.opener.requests), 240)
        self.assertEqual(Decimal(final["cumulative_reserved_usd"]), Decimal("4.92840"))
        receipt = read_json(self.output / "live_receipts" / (QUERY_RECOVERY_CELL + ".json"))
        self.assertEqual(len(receipt["sources"]), 10)
        self.assertEqual(receipt["sources"][-1]["url"], BMW_LANGUAGE_CITATION)

    def test_query_recovery_refuses_wrong_owner_digest_or_further_code_change(self):
        key, sha = self.paused_original_run()
        with self.assertRaisesRegex(LiveBlocked, "digest_mismatch"):
            self.recover("0" * 64)
        with self.assertRaisesRegex(LiveBlocked, "single_owner"):
            reconcile_query(self.input, self.output, self.access, "remaining",
                            cell=QUERY_RECOVERY_CELL, retained_sha256=sha, owner_task_id="old-task")
        self.assertEqual(Ledger(self.output / "live_journal.jsonl", "10.00").states[key], "uncertain")
        self.recover(sha)
        with patch("experiments.provider_eval_recovery.live_runner.code_hash", return_value="changed-again"):
            with self.assertRaisesRegex(LiveBlocked, "patch_receipt"):
                self.invoke("remaining", execute=True)
        self.assertEqual(len(self.opener.requests), 112)

    def test_query_patch_cannot_change_other_scope_fields(self):
        _, sha = self.paused_original_run()
        self.access["verified_model"] = "another-model"
        with self.assertRaises(LiveBlocked):
            self.recover(sha)
        self.assertEqual(len(self.opener.requests), 112)

    def test_bmw_locale_query_preserved_but_credentials_redirects_and_userinfo_refused(self):
        raw = {"results": [{"url": BMW_LANGUAGE_CITATION, "title": "BMW release",
                            "excerpts": ["Public article"]}]}
        self.assertEqual(normalize("parallel_advanced", raw)[0]["url"], BMW_LANGUAGE_CITATION)
        for url in (BMW_ARTICLE + "?token=PRIVATE", BMW_ARTICLE + "?language=PRIVATE_TOKEN",
                    BMW_ARTICLE + "?language=en&token=PRIVATE", BMW_ARTICLE + "?language=en&language=de",
                    BMW_ARTICLE + "?redirect=https://private.invalid", BMW_ARTICLE + "?language=en#PRIVATE",
                    BMW_ARTICLE.replace("https://", "https://user:password@") + "?language=en",
                    BMW_ARTICLE.replace("www.press.bmwgroup.com", "www.press.bmwgroup.com.evil") + "?language=en"):
            with self.subTest(url=url):
                raw["results"][0]["url"] = url
                with self.assertRaises(ValueError):
                    normalize("parallel_advanced", raw)


if __name__ == "__main__":
    unittest.main()
