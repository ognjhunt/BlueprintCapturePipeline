import hashlib
from pathlib import Path
import tempfile
import unittest

from .harness import ROOT
from .public_inputs import DECLARED_ORIGINAL_SHA256, load_public, prepare


class ReviewedPublicInputs(unittest.TestCase):
    def test_exact_original_public_bytes_and_all_twenty_case_prompts(self):
        path = ROOT / "real_public/inputs.parent-message.json"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), DECLARED_ORIGINAL_SHA256)
        original, cases, provenance = load_public(path)
        self.assertTrue(provenance["byte_parity_with_original_public"])
        self.assertEqual(len(cases), 20)
        for source, mapped in zip(original["cases"], cases):
            self.assertEqual(mapped["case_id"], source["id"])
            self.assertEqual(mapped["prompt"], original["common_prompt"] + "\nEntity: "
                             + source["entity"] + "\nQuestion: " + source["question"])
            self.assertEqual(mapped["search_queries"][0], source["question"])

    def test_prepare_only_real_public_requests_no_oracle_or_provider_call(self):
        with tempfile.TemporaryDirectory() as directory:
            result = prepare(ROOT / "real_public/inputs.parent-message.json", Path(directory))
            self.assertEqual(result["prepared_request_count"], 80)
            self.assertEqual(result["real_public_cases_integrated"], 20)
            self.assertEqual(result["live_provider_calls"], 0)
            self.assertFalse((Path(directory) / "reviewer").exists())


if __name__ == "__main__":
    unittest.main()
