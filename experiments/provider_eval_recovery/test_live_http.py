"""Mocked real HTTP seam; no network operations or real credentials."""

from decimal import Decimal
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION, require_paid_resource_admission,
)

from .adapters import MODEL, request
from .harness import ROOT, Ledger, digest, load_frozen
from .live_http import (HTTPTransport, LiveBlocked, NoRedirect, PROJECT, existing_key,
                        openai_envelope)


class FakeResponse(io.BytesIO):
    status = 200


class Opener:
    def __init__(self, raw):
        self.raw, self.requests, self.failure = raw, [], False

    def open(self, req, timeout):
        self.requests.append(req)
        if self.failure:
            raise TimeoutError("PRIVATE_ERROR_TEXT_MUST_NEVER_BE_ECHOED")
        return FakeResponse(json.dumps(self.raw).encode())


class MockHTTPContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)
        _, cases, fixtures = load_frozen(ROOT / "synthetic")
        self.case = cases[0]
        self.raw = fixtures["S01_parallel_fast"]["queries"][0][0]["raw"]
        self.plan = {"execution": "live_authorized", "budget_usd": "10.00", "model": MODEL,
                     "journal_root": str(self.output),
                     "openai_project": PROJECT, "case_count": 20, "phase": "pilot",
                     "public_inputs_sha256": digest(cases), "reviewer_spec_sha256": "pinned",
                     "tokenizer_id": "mock-only-not-live-tokenizer",
                     "case_hashes": {f"{i:02d}": digest(case) for i, case in enumerate(cases, 1)}}
        self.access = {"status": "existing_access_configured", "openai_project": PROJECT,
                       "allowed_hosts": ["api.parallel.ai", "api.perplexity.ai", "api.openai.com"],
                       "parallel_x_api_key_supported": True,
                       "tokenizer_id": "mock-only-not-live-tokenizer"}
        self.env = patch.dict(os.environ, {"PARALLEL_API_KEY": "mock-only",
                                          "PERPLEXITY_API_KEY": "mock-only",
                                          "OPENAI_API_KEY": "mock-only"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.opener = Opener(self.raw)
        self.transport = HTTPTransport(self.output, self.plan, self.access,
                                       opener=self.opener, token_counter=lambda _: 1)

    def grant(self, provider, plan=None):
        resource = "openai_api_candidate" if provider == "openai" else "evaluator_api"
        return require_paid_resource_admission(
            {"schema_version": PAID_LANE_ADMISSION_SCHEMA_VERSION, "status": "admitted",
             "resource_class": resource, "blockers": [],
             "allocation_binding_digest": digest(plan or self.plan)},
            resource_class=resource, expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)

    def parallel(self, **kwargs):
        return self.transport.send("parallel", request("parallel_fast", self.case, 0),
                                   cell="01_parallel_fast", role="search", attempt=1,
                                   grant=self.grant("parallel"), public_input=self.case, **kwargs)

    def test_correct_parallel_header_and_completed_adoption_without_resend(self):
        first = self.parallel()
        self.assertEqual(self.parallel(), first)
        self.assertEqual(len(self.opener.requests), 1)
        headers = dict(self.opener.requests[0].header_items())
        self.assertIn("X-api-key", headers)
        self.assertNotIn("Authorization", headers)
        self.assertNotIn("mock-only", (self.output / "live_journal.jsonl").read_text())

    def test_perplexity_and_openai_use_bearer_and_exact_project_model(self):
        self.opener.raw = {"results": [{"url": "https://example.com/page", "title": "page",
                                        "snippet": "text"}]}
        self.transport.send("perplexity", request("perplexity_fast", self.case, 0),
                            cell="01_perplexity_fast", role="search", attempt=1,
                            grant=self.grant("perplexity"), public_input=self.case)
        self.assertIn("Authorization", dict(self.opener.requests[-1].header_items()))
        self.opener.raw = {"model": MODEL, "output": [], "usage": {"input_tokens": 1,
                                                                   "output_tokens": 1}}
        text = self.transport.model_input("01_parallel_fast", "planner", self.case)
        self.transport.send("openai", openai_envelope(text, 1),
                            cell="01_parallel_fast", role="planner", attempt=1,
                            grant=self.grant("openai"), input_token_count=1, public_input=self.case)
        headers = dict(self.opener.requests[-1].header_items())
        self.assertEqual(headers["Openai-project"], PROJECT)
        self.assertEqual(json.loads(self.opener.requests[-1].data)["model"], MODEL)

    def test_secure_access_header_capability_missing_blocks_before_open(self):
        with self.assertRaisesRegex(LiveBlocked, "unconfirmed"):
            HTTPTransport(self.output, self.plan, {**self.access,
                          "parallel_x_api_key_supported": False}, opener=self.opener)
        self.assertEqual(self.opener.requests, [])

    def test_paid_admission_missing_or_wrong_binding_blocks_before_open(self):
        with self.assertRaises(RuntimeError):
            self.transport.send("parallel", request("parallel_fast", self.case, 0),
                                cell="01_parallel_fast", role="search", attempt=1,
                                grant=None, public_input=self.case)
        with self.assertRaises(RuntimeError):
            self.transport.send("parallel", request("parallel_fast", self.case, 0),
                                cell="01_parallel_fast", role="search", attempt=1,
                                grant=self.grant("parallel", {**self.plan, "other": True}),
                                public_input=self.case)
        self.assertEqual(self.opener.requests, [])

    def test_arbitrary_endpoint_model_or_private_payload_rejected(self):
        envelope = request("parallel_fast", self.case, 0)
        for url in ("https://api.parallel.ai.evil/v1/search", "http://api.parallel.ai/v1/search"):
            with self.assertRaisesRegex(LiveBlocked, "endpoint"):
                self.transport.send("parallel", {**envelope, "url": url},
                                    cell="01_parallel_fast", role="search", attempt=1,
                                    grant=self.grant("parallel"), public_input=self.case)
        with self.assertRaises(ValueError):
            self.transport.send("parallel", envelope, cell="01_parallel_fast", role="search",
                                attempt=1, grant=self.grant("parallel"),
                                public_input={**self.case, "oracle": "PRIVATE"})
        with self.assertRaisesRegex(LiveBlocked, "model"):
            body = openai_envelope(self.transport.model_input("01_parallel_fast", "planner",
                                                              self.case), 1)
            body["body"]["model"] = "gpt-6-sol"
            self.transport.send("openai", body, cell="01_parallel_fast", role="planner",
                                attempt=1, grant=self.grant("openai"), input_token_count=1,
                                public_input=self.case)
        self.assertEqual(self.opener.requests, [])

    def test_uncertain_timeout_retains_charge_and_never_retries_or_exposes_error(self):
        self.opener.failure = True
        with self.assertRaisesRegex(LiveBlocked, "uncertain_submission") as raised:
            self.parallel()
        self.assertNotIn("PRIVATE_ERROR", str(raised.exception))
        ledger = Ledger(self.output / "live_journal.jsonl", "10")
        self.assertEqual(ledger.exposure, Decimal("0.004125"))
        with self.assertRaisesRegex(LiveBlocked, "no_resend"):
            self.parallel()
        self.assertEqual(len(self.opener.requests), 1)

    def test_cumulative_pilot_cap_fails_before_dispatch(self):
        self.output.mkdir(exist_ok=True)
        ledger = Ledger(self.output / "live_journal.jsonl", "10")
        ledger.append("reserved", "prior-pilot", amount_usd="0.999", cell="01_parallel_fast",
                      provider="openai", role="planner")
        with self.assertRaisesRegex(LiveBlocked, "pilot_inclusive_cap"):
            self.parallel()
        self.assertEqual(len(self.opener.requests), 0)

    def test_openai_requires_pinned_tokenizer(self):
        transport = HTTPTransport(self.output, self.plan, self.access, opener=self.opener)
        with self.assertRaisesRegex(LiveBlocked, "tokenizer"):
            text = transport.model_input("01_parallel_fast", "planner", self.case)
            transport.send("openai", openai_envelope(text, 1), cell="01_parallel_fast",
                           role="planner", attempt=1, grant=self.grant("openai"),
                           input_token_count=1, public_input=self.case)
        self.assertEqual(self.opener.requests, [])

    def test_no_secret_binding_blocks_before_any_request(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(LiveBlocked, "binding_missing"):
                self.parallel()
        self.assertEqual(self.opener.requests, [])

    def test_mounted_existing_key_permissions_and_symlink_guards(self):
        path = self.output / "key"
        path.write_text("mock-file-only")
        path.chmod(0o600)
        with patch.dict(os.environ, {"PARALLEL_API_KEY": "", "PARALLEL_API_KEY_FILE": str(path)}):
            self.assertTrue(existing_key("parallel"))
            path.chmod(0o644)
            with self.assertRaisesRegex(LiveBlocked, "unsafe"):
                existing_key("parallel")
            link = self.output / "link"
            link.symlink_to(path)
            with patch.dict(os.environ, {"PARALLEL_API_KEY_FILE": str(link)}):
                with self.assertRaisesRegex(LiveBlocked, "unavailable"):
                    existing_key("parallel")

    def test_redirect_never_forwards_credentials(self):
        with self.assertRaisesRegex(LiveBlocked, "redirect_refused"):
            NoRedirect().redirect_request(None, None, None, None, None, None)

    def test_grant_cannot_be_reused_with_a_new_journal_directory(self):
        with self.assertRaisesRegex(LiveBlocked, "canonical_experiment_journal"):
            HTTPTransport(self.output / "new-root", self.plan, self.access, opener=self.opener)
        self.assertEqual(self.opener.requests, [])

    def test_openai_arbitrary_private_input_cannot_pass_case_provenance(self):
        with self.assertRaisesRegex(LiveBlocked, "provenance"):
            self.transport.send("openai", openai_envelope("PRIVATECRM_SENTINEL", 1),
                                cell="01_parallel_fast", role="planner", attempt=1,
                                grant=self.grant("openai"), input_token_count=1,
                                public_input=self.case)
        self.assertEqual(self.opener.requests, [])

    def test_output_alias_retarget_cannot_reset_cumulative_journal(self):
        canonical = self.output / "canonical"
        canonical.mkdir()
        alias = self.output / "alias"
        alias.symlink_to(canonical, target_is_directory=True)
        plan = {**self.plan, "journal_root": str(canonical)}
        transport = HTTPTransport(alias, plan, self.access, opener=self.opener)
        grant = self.grant("parallel", plan)
        envelope = request("parallel_fast", self.case, 0)
        kwargs = {"cell": "01_parallel_fast", "role": "search", "attempt": 1,
                  "grant": grant, "public_input": self.case}
        transport.send("parallel", envelope, **kwargs)
        replacement = self.output / "replacement"
        replacement.mkdir()
        alias.unlink()
        alias.symlink_to(replacement, target_is_directory=True)
        transport.send("parallel", envelope, **kwargs)
        self.assertEqual(len(self.opener.requests), 1)
        self.assertFalse((replacement / "live_journal.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
