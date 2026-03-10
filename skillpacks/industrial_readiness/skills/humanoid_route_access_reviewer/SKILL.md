# Humanoid Route Access Reviewer

Use when route access and mobility constraints need a humanoid-specific readout.

Inputs:
- `route_graph.json`
- `qualification_record.json`
- `geometry_evidence.json`

Required behavior:
- Summarize route width, choke points, route confidence, and access uncertainty.
- Flag unsupported access claims.
- Keep route-readiness language bounded to available measurements.

Do not:
- Claim safe traversal without measured support.
- Ignore low-confidence route edges.
- Convert pre-screen capture into mobility approval.

Output:
- Route-access review summary.
