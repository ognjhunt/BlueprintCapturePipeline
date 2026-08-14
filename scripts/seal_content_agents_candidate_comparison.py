#!/usr/bin/env python3
"""Seal several paid Content Agents runs into one comparison receipt.

`adp_content_agents_candidate_comparison` produces the receipt a human reviews
before any CAD backend is preferred over another, and it could be produced by no
script and by no module carrying a `main()`. So the comparison that decides which
backend a task keeps existed only for whoever opened a Python session, and
`tests/test_materializer_reachability.py` went to 74 against a budget of 73 that
only ratchets down.

Same defect as #512 (lanes), #520 (bundle modules), #523 (authority
materializers), the ArtiFixer3D input chain, and the semantic-teacher image-edit
closer, in a sixth scope.

The flag table below *is* the call -- the parser and the keyword arguments are
both built from it, and `tests/test_content_agents_candidate_comparison_cli.py`
derives the left column from the function's own signature.

`--candidates` is one JSON array rather than a flag per field. A candidate is a
nine-key mapping carrying a nested list of review frames, and the comparison
wants one per replacement slot per admitted backend; spreading that across flags
would put the operator in the business of reassembling a document the comparison
then validates as a whole. It is read as JSON rather than handed through as a
path because the materializer takes `Sequence[Mapping[str, Any]]` -- and a path
string is itself a `Sequence`, so passing it would iterate its characters and
refuse deep inside candidate normalization, blaming the evidence for what was a
bad command line.

Each element supplies the terminal evidence of one paid run:

    bundle_receipt_path        the sealed provider bundle it ran
    launch_profile_path        the profile that launched it
    allocator_result_path      what the allocator actually did
    artifact_manifest_path     what the run produced
    review_frame_paths         the frames a human looks at
    object_store_cleanup_path  proof the staged objects are gone
    teardown_manifest_path     proof the resource is gone
    provider_zero_path         proof the account is renting nothing
    webapp_sync_path           optional; the publication record

Reads retained bytes only; performs no provider mutation and rents nothing. It
seals what several runs already did, and refuses when the evidence is not there.
"""

from __future__ import annotations

from collections.abc import Sequence

from blueprint_pipeline.adp_content_agents_candidate_comparison import (
    materialize_content_agents_candidate_comparison,
)
from blueprint_pipeline.materializer_cli import Param, Step, run

STEPS: dict[str, Step] = {
    "comparison": Step(
        "Seal terminal candidate evidence from several paid runs into one receipt.",
        materialize_content_agents_candidate_comparison,
        {
            "candidates": Param(
                "--candidates",
                "JSON array of candidate specs, one per paid run. Refused unless "
                "every replacement slot carries every admitted backend.",
                required=True,
                json_file=True,
            ),
            "output_path": Param(
                "--output", "Where to write the sealed comparison.", required=True
            ),
            # Optional upstream, and optional here: the receipt stamps its own
            # time when this is omitted. Pinning it is for replaying a
            # comparison to the same bytes.
            "generated_at": Param(
                "--generated-at", "Override the receipt timestamp, for exact replay."
            ),
        },
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    return run(STEPS, argv, description=__doc__)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
