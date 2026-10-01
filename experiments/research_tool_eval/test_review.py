import copy
import json
import unittest

from harness import RAW_MODES, ROOT, GateError, run_plan, verify_freeze
from review import apply_review, make_review, quote_audit


class Grading(unittest.TestCase):
    def setUp(self):
        self.case = json.loads((ROOT / "cases.json").read_text())[0]
        self.cell = {"case_id": self.case["id"], "mode": "parallel-fast", "controller_request_id": "request1",
                     "status": "completed",
                     "sources": [{"url": "https://example.org/source", "text": "Robots need site acceptance."}],
                     "answer": {"summary": "candidate", "claims": [{"id": "claim1", "quote": "need site acceptance",
                                                                            "source_url": "https://example.org/source"}],
                                "useful_fields": {}, "usage": {"actual_usd": .001}}, "grading": {}}
        cells = []
        for case in json.loads((ROOT / "cases.json").read_text()):
            for mode in RAW_MODES:
                c = copy.deepcopy(self.cell)
                c.update(case_id=case["id"], mode=mode, controller_request_id=case["id"] + mode)
                cells.append(c)
        self.report = {"freeze": verify_freeze(), "cells": cells, "offline": False, "run_plan": run_plan()}

    def complete_review(self):
        review, mapping = make_review(self.report)
        for e in review["review"]:
            for x in e["expected_fact_grades"]:
                x.update(grade="omitted", support_note="fixture omits real facts")
            e["useful_field_grades"] = dict.fromkeys(e["useful_field_grades"], False)
            for x in e["claim_grades"]:
                x.update(supported=True, citation_correct=True, quote_supported=True, source_freshness="unknown")
            e.update(unsupported_claims=[], critical_error=False, unknown_and_conflict_handling=True,
                     freshness="unknown", reviewer="hermetic-test-reviewer", reviewed_at="2026-09-30")
        return review, mapping

    def test_quote_match_is_screen_not_semantic_grade(self):
        audit = quote_audit(self.cell["answer"], self.cell["sources"])
        self.assertTrue(audit[0]["quote_in_retained_passage"])
        self.assertEqual("pending_human_review", audit[0]["semantic_support"])
        self.cell["answer"]["claims"][0]["source_url"] = "https://example.org/unretrieved"
        self.assertFalse(quote_audit(self.cell["answer"], self.cell["sources"])[0]["quote_in_retained_passage"])

    def test_blinded_review_excludes_mode_and_usage(self):
        review, mapping = make_review(self.report)
        self.assertNotIn("parallel-fast", json.dumps(review))
        self.assertNotIn("usage", review["review"][0]["answer"])
        self.assertIn(next(iter(mapping.values()))["mode"], RAW_MODES)
        self.assertTrue(review["review"][0]["required_unknowns"])

    def test_incomplete_human_grades_refused(self):
        r, m = make_review(self.report)
        with self.assertRaises(GateError):
            apply_review(self.report, r, m)

    def test_mock_quality_rankings_refused(self):
        r, m = self.complete_review()
        self.report["offline"] = True
        with self.assertRaises(GateError):
            apply_review(self.report, r, m)

    def test_denominators_and_cell_mapping_cannot_change(self):
        r, m = self.complete_review()
        r["review"][0]["expected_fact_grades"].pop()
        with self.assertRaises(GateError):
            apply_review(self.report, r, m)
        r, m = self.complete_review()
        next(iter(m.values()))["controller_request_id"] = "different"
        with self.assertRaises(GateError):
            apply_review(self.report, r, m)

    def test_complete_labels_attach_evidence_metrics(self):
        r, m = self.complete_review()
        result = apply_review(copy.deepcopy(self.report), r, m)
        grade = result["cells"][0]["grading"]
        self.assertEqual("human_reviewed", grade["state"])
        self.assertEqual(0, grade["coverage"])
        self.assertEqual(1, grade["citation_support"])
        self.assertEqual([], grade["useful_accepted_fields"])

    def test_missing_or_one_cell_report_cannot_claim_twenty_case_results(self):
        r, m = self.complete_review()
        self.report["cells"] = self.report["cells"][:1]
        r2, m2 = make_review(self.report)
        with self.assertRaises(GateError):
            apply_review(self.report, r2, m2)
        with self.assertRaises(GateError):
            apply_review(self.report, r, m)

    def test_answer_or_sources_changed_after_grading_refused(self):
        r, m = self.complete_review()
        original = copy.deepcopy(self.report)
        for field in ("answer", "sources"):
            changed = copy.deepcopy(original)
            if field == "answer":
                changed["cells"][0]["answer"]["summary"] = "tampered without changing claim IDs"
            else:
                changed["cells"][0]["sources"][0]["text"] = "tampered source"
            with self.assertRaises(GateError):
                apply_review(changed, r, m)

    def test_expected_facts_cannot_change_during_review(self):
        r, m = self.complete_review()
        r["review"][0]["expected_fact_grades"][0]["statement"] = "different truth"
        with self.assertRaises(GateError):
            apply_review(self.report, r, m)

    def test_failed_cells_remain_and_cannot_earn_credit(self):
        self.report["cells"][0].update(status="failed", answer=None, controller_request_id=None)
        r, m = self.complete_review()
        result = apply_review(copy.deepcopy(self.report), r, m)
        self.assertEqual(80, len(result["cells"]))
        self.assertEqual(0, result["cells"][0]["grading"]["coverage"])
        failed = next(x for x in r["review"] if x["status"] == "failed")
        failed["expected_fact_grades"][0]["grade"] = "supported"
        with self.assertRaises(GateError):
            apply_review(self.report, r, m)


if __name__ == "__main__":
    unittest.main()
