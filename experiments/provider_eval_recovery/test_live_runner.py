"""Hermetic end-to-end live pilot tests. All HTTP and keys are mocked."""

from decimal import Decimal
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .adapters import MODEL
from .harness import ROOT, Ledger, read_json
from .live_http import COUNT_ENDPOINT, COUNT_METHOD, LiveBlocked, PROJECT
from .live_runner import APPROVAL, CATALOG, OWNER, run


class Response(io.BytesIO):
    status = 200


class MockProviders:
    def __init__(self):
        self.requests = []
        self.count = 500
        self.fail = False

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
            raw = {"results": [result]}
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


if __name__ == "__main__":
    unittest.main()
