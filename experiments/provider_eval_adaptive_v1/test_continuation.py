"""Mock-only citation validation and accepted-search continuation regressions."""

from decimal import Decimal
import json
import unittest
from unittest.mock import patch

from experiments.provider_eval_recovery.harness import Ledger, digest, read_json, write_once
from experiments.provider_eval_recovery.live_http import COUNT_ENDPOINT

from . import test_adaptive as fixtures
from .citations import CitationNormalizationError, ProviderInputWarning, audit_urls, normalize_public
from .continuation import BASE_CODE, CELL, RECEIPT_NAME, ContinuationError, receipt_path
from .protocol import CURRENT_DATE, MODES, PROTOCOL, SEEDS, evidence, search_request
from .reviewer import bind_reviewer, review_case
from .runner import Blocked, UNUSED_REVIEWED_CODE, active_scope, admission, run


class CitationContinuationTests(unittest.TestCase):
    setUp = fixtures.AdaptiveTests.setUp
    transport = fixtures.AdaptiveTests.transport
    invoke = fixtures.AdaptiveTests.invoke

    def seed_accepted_failure(self, *, warning=False):
        current, public, paths = admission(self.root, self.access, self.inputs)
        original = {**current, "code_sha256": UNUSED_REVIEWED_CODE}
        base = {**current, "code_sha256": BASE_CODE}
        (paths / "scope.json").write_text(json.dumps(original))
        write_once(paths / "unused_scope_adoption.json", {"original_scope_sha256": digest(original),
            "reviewed_original_commit": "63ad3659112b2e9befd2b9fcffdf03160f22562a",
            "zero_adaptive_reservations_verified": True, "adopted_scope": base})
        request = search_request(1, "parallel_fast", public["cases"][0], SEEDS[0])
        key = digest({"plan": digest(base), "cell": CELL, "step": "search1", "request": request})
        # Illustrative safe URLs and synthetic excerpts, not the live source list.
        urls = [f"https://www.chefrobotics.ai/?3f5a3c8b_page={page}" for page in (2, 3, 12)]
        urls += [f"https://example.com/public-{i}" for i in range(7)]
        raw = {"warnings": [{"type": "input_validation"}] if warning else None,
               "results": [{"url": url, "title": "Chef Robotics synthetic public fixture",
                            "excerpts": ["Synthetic task evidence; suitability unknown."]} for url in urls]}
        retained = {"request": request, "raw": raw, "latency_seconds": 1.009,
                    "billing": "unreconciled; full reservation retained"}
        write_once(paths / "raw" / (key + ".json"), retained)
        ledger = Ledger(self.root / "live_journal.jsonl", "10.00")
        ledger.append("reserved", key, amount_usd="0.004125", cell=CELL, provider="parallel", step="search1",
                      role=PROTOCOL + ":search1", protocol=PROTOCOL, plan_sha256=digest(base), request_sha256=digest(request))
        ledger.append("completed", key, raw_sha256=digest(retained))
        failure = {"protocol": PROTOCOL, "case_id": "BP-EVAL-01", "cell": CELL, "mode": "parallel_fast",
            "status": "unknown_contract_or_evidence_stop_not_provider_quality",
            "answer": "Unknown: no supported answer within this protocol.", "sources": [], "operational_coverage": None,
            "grade": "pending isolated reviewer; controller triage is not ground truth",
            "retained_attempts": {key: digest(retained)}, "current_date": CURRENT_DATE}
        write_once(paths / "receipts" / (CELL + ".json"), failure)
        return base, paths, key, digest(retained)

    def continue_free(self, key, sha, **kwargs):
        scope = kwargs.pop("expected_adaptive_scope_sha256", digest(active_scope(self.root / "protocols" / PROTOCOL)))
        return self.invoke(continue_retained_attempt=key, retained_envelope_sha256=sha,
                           expected_adaptive_scope_sha256=scope, **kwargs)

    def test_confirmed_pagination_preserves_full_url_and_bmw_locale(self):
        chef = "https://www.chefrobotics.ai/?3f5a3c8b_page=12"
        bmw = ("https://www.press.bmwgroup.com/global/article/detail/T0458778EN/"
               "bmw-group-advances-the-use-of-physical-ai-in-production-with-figure-03-project-in-spartanburg?language=en")
        for mode in MODES:
            raw = {"results": [{"url": url, "title": "Public", "excerpts": ["Bounded evidence"],
                               "snippet": "Bounded evidence"} for url in (chef, bmw)]}
            self.assertEqual([s["url"] for s in normalize_public(mode, raw)], [chef, bmw])
            self.assertEqual(len(audit_urls(mode, raw)), 2)
            for chars in (1000, 5500):
                sources = evidence(mode, [raw], chars)
                self.assertLessEqual(sum(len(s["title"]) + len(s["url"]) + len(s["text"]) for s in sources), chars)

    def test_query_userinfo_fragment_redirect_and_credential_guards_remain(self):
        invalid = ("https://user:password@www.chefrobotics.ai/blog?3f5a3c8b_page=2",
                   "https://www.chefrobotics.ai/blog?3f5a3c8b_page=2#private",
                   "https://www.chefrobotics.ai/blog?3f5a3c8b_page=-2",
                   "https://www.chefrobotics.ai/blog?3f5a3c8b_page=secret",
                   "https://www.chefrobotics.ai/blog?3f5a3c8b_page=1234567",
                   "https://www.chefrobotics.ai/blog?3f5a3c8b_page=2&redirect=https://example.com",
                   "https://www.chefrobotics.ai/blog?token=secret",
                   "https://www.chefrobotics.ai/blog?3f5a3c8b_page=2&3f5a3c8b_page=3",
                   "https://different.example/blog?3f5a3c8b_page=2")
        for url in invalid:
            with self.assertRaises(CitationNormalizationError):
                audit_urls("parallel_fast", {"results": [{"url": url}]})

    def test_url_audit_catches_invalid_last_url_even_after_context_clip(self):
        raw = {"results": [{"url": "https://example.com/public", "title": "Public", "excerpts": ["X" * 6000]},
                           {"url": "https://example.com/private?token=secret", "title": "", "excerpts": []}]}
        with self.assertRaises(CitationNormalizationError):
            normalize_public("parallel_fast", raw)

    def test_free_continuation_and_pilot_replay_preserve_search_scope_journal_and_failure(self):
        base, paths, key, sha = self.seed_accepted_failure()
        preserved = {path: path.read_bytes() for path in (paths / "scope.json", paths / "unused_scope_adoption.json",
                     paths / "receipts" / (CELL + ".json"), paths / "raw" / (key + ".json"))}
        prefix = (self.root / "live_journal.jsonl").read_bytes()
        for _ in range(2):
            result = self.continue_free(key, sha)
            self.assertEqual(result["accepted_search_continuation"]["citation_urls_audited"], 10)
        self.assertEqual(self.opener.requests, [])
        self.assertEqual((self.root / "live_journal.jsonl").read_bytes(), prefix)
        self.assertEqual(active_scope(paths), base)
        self.assertEqual(Ledger(self.root / "live_journal.jsonl", "10.00").exposure, Decimal("2.289635"))
        output = self.invoke(True)
        self.assertEqual(self.opener.requests[0].full_url, COUNT_ENDPOINT)
        self.assertEqual(len(self.opener.requests), 47)  # retained search adopted; never a 48th request
        self.assertEqual(Decimal(output["aggregate_reserved_usd"]), Decimal("2.931750"))
        self.invoke(True)
        self.assertEqual(len(self.opener.requests), 47)
        continued = read_json(receipt_path(paths, CELL))
        self.assertEqual(continued["status"], "completed_ungraded")
        self.assertIn("3f5a3c8b_page=2", json.dumps(continued["sources"]))
        self.assertTrue((self.root / "live_journal.jsonl").read_bytes().startswith(prefix))
        for path, blob in preserved.items():
            self.assertEqual(path.read_bytes(), blob)
        reviewer = self.transport(True)
        bundle = {"spec": "SYNTHETIC_REVIEW_ONLY", "cases": {case["id"]: "synthetic" for case in reviewer.public["cases"]}}
        bind_reviewer(reviewer, bundle)
        review_case(reviewer, 1, "synthetic", bundle["spec"])
        self.assertEqual(len(self.opener.requests), 49)

    def test_continuation_refuses_wrong_owner_attempt_digest_or_execute(self):
        _, paths, key, sha = self.seed_accepted_failure()
        prefix = (self.root / "live_journal.jsonl").read_bytes()
        for attempt, digest_value in (("0" * 64, sha), (key, "0" * 64)):
            with self.assertRaises((Blocked, ContinuationError)):
                self.continue_free(attempt, digest_value)
        with self.assertRaises(Blocked):
            run(self.root, self.access, self.inputs, continue_retained_attempt=key,
                retained_envelope_sha256=sha, owner_task_id="other")
        with self.assertRaises(Blocked):
            self.continue_free(key, sha, execute=True)
        with self.assertRaises(Blocked):
            self.continue_free(key, sha, expected_adaptive_scope_sha256="0" * 64)
        self.assertFalse((paths / RECEIPT_NAME).exists())
        self.assertEqual((self.root / "live_journal.jsonl").read_bytes(), prefix)
        self.assertEqual(self.opener.requests, [])

    def test_continuation_refuses_any_additional_adaptive_attempt(self):
        base, paths, key, sha = self.seed_accepted_failure()
        ledger = Ledger(self.root / "live_journal.jsonl", "10.00")
        ledger.append("reserved", "additional-attempt", amount_usd="0.02", protocol=PROTOCOL, plan_sha256=digest(base))
        ledger.append("not_accepted", "additional-attempt", nonacceptance_proof="synthetic-only")
        with self.assertRaisesRegex(Blocked, "only_accepted_search_attempt"):
            self.continue_free(key, sha)
        self.assertFalse((paths / RECEIPT_NAME).exists())
        self.assertEqual(self.opener.requests, [])

    def test_continuation_never_relabels_provider_warning_as_local_fix(self):
        _, paths, key, sha = self.seed_accepted_failure(warning=True)
        with self.assertRaises(ProviderInputWarning):
            self.continue_free(key, sha)
        self.assertFalse((paths / RECEIPT_NAME).exists())
        self.assertEqual(self.opener.requests, [])

    def test_changed_retained_response_or_patch_receipt_blocks_before_paid_stage(self):
        _, paths, key, sha = self.seed_accepted_failure()
        self.continue_free(key, sha)
        patch_path = paths / RECEIPT_NAME
        correct = read_json(patch_path)
        patch_path.write_text(json.dumps({**correct, "patched_code_sha256": "0" * 64}))
        with self.assertRaisesRegex(Blocked, "integrity_failure"):
            self.invoke(True)
        patch_path.write_text(json.dumps(correct))
        raw_path = paths / "raw" / (key + ".json")
        raw = read_json(raw_path)
        raw["raw"]["results"][0]["title"] = "Altered response"
        raw_path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(Blocked, "digest_failure"):
            self.invoke(True)
        self.assertEqual(self.opener.requests, [])

    def test_local_citation_failure_is_distinct_from_provider_warning_and_stops_phase(self):
        original_open = self.opener.open
        def unsafe_source(req, timeout):
            response = original_open(req, timeout)
            raw = json.loads(response.getvalue())
            raw["results"][0]["url"] = "https://example.com/public?token=secret"
            return fixtures.Response(json.dumps(raw).encode())
        with patch.object(self.opener, "open", unsafe_source):
            with self.assertRaisesRegex(Blocked, "local_citation_normalizer"):
                self.invoke(True)
        path = self.root / "protocols" / PROTOCOL / "receipts" / (CELL + ".json")
        self.assertTrue(read_json(path)["status"].startswith("unknown_local_citation_normalization"))
        self.assertEqual(len(self.opener.requests), 1)


if __name__ == "__main__":
    unittest.main()
