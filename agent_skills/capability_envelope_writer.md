# Capability Envelope Writer

Input:
- `scene_graph.json`
- `route_graph.json`
- `geometry_evidence.json`
- `task_scope_record.json`

Task:
- Write bounded locomotion, reach, occupancy, choke-point, occlusion, and route-viability statements from structured evidence.
- Do not infer pass/fail on unsupported geometry.

Output:
- Structured capability-check language only.
