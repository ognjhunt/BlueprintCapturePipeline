# NVIDIA USD Content Agents comparison: closed with a result

**Status:** closed, 2026-08-18. Bounded comparison complete; no further paid
Joint Agent work is planned for Scene 840920.

## What was compared

NVIDIA USD Content Agents v0.5.2 (commit `36dbf3f2`), Joint Rigger app, run
against Scene 840920 Task A's washer. The question was narrow and was never
whether the tool could qualify anything: its output is capped at
`joint_agent_output_is_optional_topology_candidate` everywhere it appears.
The question was whether an outside tool, given a fair input, proposes the
articulation we authored.

## Result

On 2026-08-18 the pipeline completed all eight stages for the first time --
`optimize_usd`, `identify_asset`, `analyze_structure`, `build_dataset_usd`,
`build_dataset_prepare_dataset`, `predict`, `consistency_pass`,
`infer_articulation_candidates` -- and emitted eight articulation candidates
(run `adp-joint-agent-840920-task-a-000d9b9b-composed-api-20260818T134451Z-4e8c31b0`,
$0.0747, torn down, provider zero confirmed).

**Two candidates resolved a parent, and both match the frozen graph:**

| Agent candidate | Frozen graph joint | Agreement |
| --- | --- | --- |
| `drum` revolute on `body` | `drum_bearing`, revolute, drum-to-body, role `passive` | motion type and parent |
| `drawer` prismatic on `body` | `drawer_slide`, prismatic, drawer-to-body, role `locked` | motion type and parent |

Both carry `parent_resolution_source: stage1_rigger_evidence`.

**Six candidates did not resolve a parent.** Each names a visual sub-mesh
rather than the link that owns it -- for example
`/Asset/links/door/visuals/door__door_mesh` instead of `/Asset/links/door` --
and reports `fixed_parent_prim: None`, `parent_resolution_source: unresolved`.
The deterministic review therefore refused the run on
`exactly_one_task_joint_not_resolved` and
`joint_candidate_count_outside_preregistered_bounds`.

## Reading the result honestly

The comparison corroborates the passive and locked joints of the frozen graph
and does not resolve the commanded one. That is a real, bounded finding and it
is the comparison's answer.

The six unresolved candidates are **our composition's shape, not the tool's
error**. Composition attaches agent-authored CAD visuals as children of each
link, and a pipeline that identifies and predicts from rendered views names
the geometry it can see. Given a mesh hierarchy whose visible leaves are one
level below the links, naming the leaf is the reasonable behaviour.

Three inputs were tried, and each failure was ours:

1. **Source mesh** (2026-08-17): parts were not prims, so no parent could be
   named. Zero candidates.
2. **Authored replacement** (2026-08-18): links were prims but collision-only,
   `purpose=guide` and invisible, so the renderer produced nothing -- the log
   ends at `No valid world-space bbox for prim /`. Pipeline aborted at
   `identify_asset`.
3. **Composed asset** (2026-08-18): articulated *and* rendered. Pipeline
   completed; two joints corroborated; six named sub-meshes.

## Why this does not affect delivery

The Joint lane is optional by construction and every run says so in its own
bundle receipt:

```
execution_role: "optional_construction_enrichment"
failure_blocks_deterministic_asset_construction: false
failure_blocks_native_simulator_qualification: false
```

No module in the delivery chain -- paired-target preflight, visual
composition, graph-asset authoring, or the Isaac probe lane -- reads Joint
Agent output. Empirically: Joint blocked on every attempt while the washer
imported cleanly into Isaac and its door was measured through a commanded
sweep to 54.99 degrees on the same day.

The commanded joint's physics is established by measurement and simulation,
not by this comparison:

- the measured derivation computes the door hinge axis from the frozen scan
  (clearance rule, `axis [0, 0, -1]`);
- the Isaac probe read that axis back physically across 0/15/30/45/55 degrees.

## Known repair, if the family is ever reopened

Fold visual descendants to their owning link before matching -- either
collapse per-link visuals into a single mesh during composition, or map
`/Asset/links/<link>/visuals/...` to `/Asset/links/<link>` in the
deterministic review. This is mechanical. It would test our composition
layout rather than the tool, which is why it was not spent on here.
