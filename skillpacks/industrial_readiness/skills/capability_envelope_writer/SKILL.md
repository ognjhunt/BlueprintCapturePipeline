# Capability Envelope Writer

Use when capability claims need to be bounded to measured evidence from the qualification pipeline.

Inputs:
- `scene_graph.json`
- `route_graph.json`
- `geometry_evidence.json`
- `task_scope_record.json`
- `capability_checks.json`

Required behavior:
- Write bounded locomotion, reach, occupancy, choke-point, occlusion, and route-viability statements.
- Keep every claim traceable to measured or explicitly declared evidence.
- Mark unsupported capability areas as unresolved.

Do not:
- Infer pass/fail on unsupported geometry.
- Invent measurements.
- Convert bounded evidence into deployment approval.

Output:
- Structured capability-envelope language only.
