# Evidence Auditor

Input:
- `capture_qa_scorecard.json`
- `geometry_evidence.json`
- `scene_graph.json`
- `route_graph.json`

Task:
- Check that every readiness claim links to a structured artifact.
- Call out missing geometry, hidden zones, low-confidence route edges, and unsupported assumptions.

Output:
- Evidence gaps and recommended escalation only.
