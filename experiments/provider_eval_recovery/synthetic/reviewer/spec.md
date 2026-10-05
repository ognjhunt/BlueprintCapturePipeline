# Isolated synthetic rubric

This is a scaffold for offline lifecycle verification. All companies, pages, provider results, model answers, token counts and latency values are invented. No provider result is measured.

Per claim, report supported/contradicted/unknown accuracy, citation resolution, primary-source status, excerpt entailment, unsupported assertions and appropriate unknowns. Synthetic entailment requires the exact pinned passage from the independently held oracle and its URL. Missing information is unknown; never infer site safety or readiness from advertised capability. A source URL alone does not prove a claim.

For the real bundle, use its reviewer/spec.md unchanged. An isolated GPT6.1Sol reviewer sees the oracle plus retained provider output, never participates in provider queries, and checks claim-to-source entailment, company/capability identity, recency and unknowns. Retain disagreements and evidence references. Unknowns stay in the denominator. Blind mode identity before judging. Grading cannot fetch live pages in this offline phase.
