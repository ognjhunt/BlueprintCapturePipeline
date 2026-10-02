"""Integrate the exact provider-safe parent-message input, without oracle access."""

import argparse
import hashlib
import json
from pathlib import Path

from .adapters import MODES, public_case, request
from .harness import ROOT, digest, write_once

DECLARED_ORIGINAL_SHA256 = "cac8d7a31aea2ad1c2e5a47ea37e434abcaae4b4910ba4afc8f81dd11c404ff3"


def load_public(path):
    blob = Path(path).read_bytes()
    data = json.loads(blob)
    expected = {"version", "as_of", "common_prompt", "cases", "case_kind",
                "hypothetical_site_scenarios"}
    if (set(data) != expected or data["version"] != "blueprint-provider-eval-inputs-v1"
            or data["as_of"] != "2026-09-30" or data["hypothetical_site_scenarios"] is not True
            or len(data["cases"]) != 20 or not isinstance(data["common_prompt"], str)):
        raise ValueError("reviewed parent public-input schema mismatch")
    mapped = []
    for index, case in enumerate(data["cases"], 1):
        if (set(case) != {"id", "entity", "question"} or case["id"] != f"BP-EVAL-{index:02d}"
                or any(not isinstance(case[key], str) or not case[key].strip()
                       for key in ("entity", "question"))):
            raise ValueError("case order, identity or public fields mismatch")
        # Preserve the entire common prompt and exact entity/question. Explicit
        # wrappers add metadata, not alternate claims or research instructions.
        prompt = (data["common_prompt"] + "\nEntity: " + case["entity"]
                  + "\nQuestion: " + case["question"])
        mapped.append(public_case({"case_id": case["id"], "prompt": prompt,
                                   "search_queries": [case["question"]] * 3,
                                   "hypothetical_scenario": "All site scenarios in the question are"
                                   " explicitly hypothetical; do not qualify a site."}))
    return data, mapped, {"route": "parent_message_provider_safe_input",
                         "source_thread_id": "01a0ef70-d046-74f6-9434-a19e5456b0ef",
                         "declared_original_public_sha256": DECLARED_ORIGINAL_SHA256,
                         "local_message_bytes_sha256": hashlib.sha256(blob).hexdigest(),
                         "byte_parity_with_original_public": hashlib.sha256(blob).hexdigest()
                         == DECLARED_ORIGINAL_SHA256,
                         "semantic_sha256": digest(data), "real_public_cases_integrated": 20,
                         "original_zip_sha256": "294f778b32670ee6ae07412a53c1627923f1004afeea501f32d3202e8b9cf736",
                         "library_materialization": "still_blocked_after_two_supported_attempts",
                         "original_zip_hash_verified_in_executor": False,
                         "oracle": "parent_side_only_not_loaded_by_controller"}


def prepare(path, output):
    data, cases, provenance = load_public(path)
    write_once(Path(output) / "public_provenance.json", provenance)
    write_once(Path(output) / "public_mapped.json", {"cases": cases})
    envelopes = {case["case_id"] + "_" + mode: request(mode, case, 0)
                 for case in cases for mode in MODES}
    write_once(Path(output) / "requests.json", envelopes)
    # Canonical round-trip verifies no supplied case text was lost or modified.
    for original, mapped in zip(data["cases"], cases):
        assert original["question"] in mapped["prompt"]
        assert original["entity"] in mapped["prompt"]
        assert data["common_prompt"] in mapped["prompt"]
    return {**provenance, "prepared_request_count": len(envelopes), "live_provider_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "real_public/inputs.parent-message.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
